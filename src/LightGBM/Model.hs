module LightGBM.Model
  ( -- * Models
    Model
  , trainNewModel
  , loadModelFromFile
    -- * Prediction
  , predict
  ) where

import           Numeric.Natural (Natural)

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
     FilePath -- ^ Where to save the new model
  -> [P.Param] -- ^ Training parameters
  -> DS.DataSet -- ^ Training data
  -> DS.DataSet -- ^ Testing data
  -> Natural -- ^ Number of training rounds
  -> IO (Either CLW.ErrLog Model)
trainNewModel modelOutputPath trainingParams trainingData validationData numRounds = do
  let dataParams = [P.Header (DS.getHeader . DS.hasHeader $ trainingData)]
      runParams =
        [ P.Task P.Train
        , P.TrainingData (DS.dataPath trainingData)
        , P.ValidationData (DS.dataPath validationData)
        , P.Iterations numRounds
        , P.OutputModel modelOutputPath
        ]
  runlog <- CLW.run lightgbmExe $ concat [runParams, trainingParams, dataParams]
  return $ either Left (\_ -> Right $ Model modelOutputPath) runlog

-- | Persisted models can be loaded up and used for prediction.
loadModelFromFile :: FilePath -> Model
loadModelFromFile = Model

-- | Predict the results of new inputs and persist the results to an
-- output file.
predict ::
     Model -- ^ A model to do prediction with
  -> DS.DataSet -- ^ The new input data for prediction
  -> FilePath -- ^ Where to persist the prediction outputs
  -> IO DS.DataSet -- ^ The prediction output DataSet
predict model inputData predictionOutputPath = do
  let dataParams = [P.Header (DS.getHeader . DS.hasHeader $ inputData)]
      runParams =
        [ P.Task P.Predict
        , P.InputModel $ modelPath model
        , P.PredictionData $ DS.dataPath inputData
        , P.OutputResult predictionOutputPath
        ]
  _ <- CLW.run lightgbmExe $ concat [dataParams, runParams]
  return $ DS.DataSet predictionOutputPath (DS.HasHeader False)
