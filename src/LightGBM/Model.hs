{-# LANGUAGE RecordWildCards #-}

module LightGBM.Model
  ( -- * Models
    Model
  , trainNewModel
  , readModelFile
  , writeModelFile
    -- * Prediction
  , predict
  ) where

import           Data.List (find)
import           System.Directory (copyFile)
import           System.IO.Temp (emptySystemTempFile)

import qualified LightGBM.DataSet as DS
import qualified LightGBM.Internal.CommandLineWrapper as CLW
import qualified LightGBM.Internal.CLIParameters as CLIP
import qualified LightGBM.Parameters as P
import           LightGBM.Utils.Types (ErrLog)

-- | A model to use to make predictions
data Model = Model
  { modelPath :: FilePath
  } deriving (Eq, Show)

lightgbmExe :: String
lightgbmExe = "lightgbm"

-- | Train a new model and persist it to a file.
trainNewModel ::
     [P.Param] -- ^ Training parameters
  -> DS.DataSet -- ^ Training data
  -> [DS.DataSet] -- ^ Testing data
  -> IO (Either ErrLog Model)
trainNewModel trainingParams trainingData validationData = do
  modelOutputPath <- getModelOutputPath
  let dataParams = [CLIP.Header (DS.getHeader . DS.hasHeader $ trainingData)]
      taskParams = [CLIP.Task CLIP.Train]
      runParams =
        [ P.TrainingData (DS.dataPath trainingData)
        , P.ValidationData $ fmap DS.dataPath validationData
        ] ++
        if hasModelOutputPathParam
          then []
          else [P.OutputModel modelOutputPath]
  runlog <-
    CLW.run lightgbmExe (runParams ++ trainingParams) [] (dataParams ++ taskParams)
  return $ either Left (\_ -> Right $ Model modelOutputPath) runlog
  where
    isOutputModelParam (P.OutputModel _) = True
    isOutputModelParam _ = False
    hasModelOutputPathParam =
      case filter isOutputModelParam trainingParams of
        [] -> False
        _ -> True
    getModelOutputPath =
      case find isOutputModelParam trainingParams of
        Just (P.OutputModel path) -> return path
        _ -> emptySystemTempFile "modelOutput"


-- | Models can be written out to a file
writeModelFile :: FilePath -> Model -> IO ()
writeModelFile outPath Model {..} = copyFile modelPath outPath

-- | Persisted models can be loaded up and used for prediction.
readModelFile :: FilePath -> IO Model
readModelFile = return . Model

-- | Predict the results of new inputs and persist the results to an
-- output file.
predict ::
     Model -- ^ A model to do prediction with
  -> [P.Param] -- ^ Parameters
  -> [P.PredictionParam]
  -> DS.DataSet -- ^ The new input data for prediction
  -> IO (Either ErrLog DS.DataSet) -- ^ The prediction output DataSet
predict model genericParams predParams inputData = do
  predictionOutputPath <- getOutputPath genericParams
  let dataParams = [CLIP.Header (DS.getHeader . DS.hasHeader $ inputData)]
      taskParams = [CLIP.Task CLIP.Predict]
      runParams =
        [ P.InputModel $ modelPath model
        , P.PredictionData $ DS.dataPath inputData
        ] ++
        if hasOutputParam genericParams
          then []
          else [P.OutputResult predictionOutputPath]
  runResults <-
    CLW.run
      lightgbmExe
      (genericParams ++ runParams)
      predParams
      (dataParams ++ taskParams)
  return $
    either
      Left
      (\_ -> Right $ DS.CSVFile predictionOutputPath (DS.HasHeader False))
      runResults
  where
    isOutputParam :: P.Param -> Bool
    isOutputParam p =
      case p of
        (P.OutputResult _) -> True
        _ -> False
    hasOutputParam :: [P.Param] -> Bool
    hasOutputParam ps =
      case filter isOutputParam ps of
        [] -> False
        _ -> True
    getOutputPath :: Foldable t => t P.Param -> IO FilePath
    getOutputPath ps =
      case find isOutputParam ps of
        Just (P.OutputResult path) -> return path
        _ -> emptySystemTempFile "predictionOutput"
