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
import qualified LightGBM.Parameters as P

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
  -> IO (Either CLW.ErrLog Model)
trainNewModel trainingParams trainingData validationData = do
  modelOutputPath <- getModelOutputPath
  let dataParams = [P.Header (DS.getHeader . DS.hasHeader $ trainingData)]
      runParams =
        [ P.Task P.Train
        , P.TrainingData (DS.dataPath trainingData)
        , P.ValidationData $ fmap DS.dataPath validationData
        ] ++
        if hasModelOutputPathParam
          then []
          else [P.OutputModel modelOutputPath]
  runlog <- CLW.run lightgbmExe $ concat [runParams, trainingParams, dataParams]
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
  -> DS.DataSet -- ^ The new input data for prediction
  -> IO DS.DataSet -- ^ The prediction output DataSet
predict model inputData = do
  predictionOutputPath <- emptySystemTempFile "predictionOutput"
  let dataParams = [P.Header (DS.getHeader . DS.hasHeader $ inputData)]
      runParams =
        [ P.Task P.Predict
        , P.InputModel $ modelPath model
        , P.PredictionData $ DS.dataPath inputData
        , P.OutputResult predictionOutputPath
        ]
  -- FIXME Handle the error case properly
  _ <- CLW.run lightgbmExe $ concat [dataParams, runParams]
  return $ DS.CSVFile predictionOutputPath (DS.HasHeader False)
