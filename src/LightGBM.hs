-- | A simple wrapper around the
-- <http://lightgbm.readthedocs.io/en/latest/index.html Microsoft LightGBM library>.
-- Documentation for the various library parameters (see
-- "LightGBM.Parameters") can be found
-- <https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst here>.
--
-- /N.B.  The 'lightgbm' executable must be on the system 'PATH'./
--
-- __/N.B. This is alpha-level software and should not be/__
-- __/used in production since the API may still change substantially./__
--
-- The basic usage of the library looks something like this:
--
-- >  import           LightGBM (loadDataFromFile, HasHeader(..), trainNewModel)
-- >  import qualified LightGBM.Parameters as P
-- >  import           Refined (refineTH)
-- >
-- >  let modelFile = "/path/to/model/output"
-- >      trainingData = loadDataFromFile (HasHeader False) "/path/to/training/data"
-- >      validationData = loadDataFromFile (HasHeader False) "/path/to/validation/data"
-- >      trainingParams = [ P.App P.Binary
-- >                       , P.Metric [P.BinaryLogloss, P.AUC]
-- >                       , P.TrainingMetric True
-- >                       , P.LearningRate 0.1
-- >                       , P.NumLeaves 63
-- >                       , P.FeatureFraction $$(refineTH 0.8)
-- >                       , P.BaggingFreq $$(refineTH 5)
-- >                       , P.BaggingFraction $$(refineTH 0.8)
-- >                       , P.MinDataInLeaf 50
-- >                       , P.MinSumHessianInLeaf 5.0
-- >                       , P.IsSparse True
-- >                       ]
-- >  model <- trainNewModel modelFile trainingParams trainingData validationData 100
-- >
-- >  let newData = loadDataFromFile (HasHeader False) "/path/to/inputs_for_prediction"
-- >      outputFile = "/path/to/prediction_outputs"
-- >  predict model newData outputFile
--
-- Note that in current versions of LightGBM, categorical features
-- must be encoded as 'Int's.
--
module LightGBM
  ( -- * Data Handling
    loadDataFromFile
  , DataSet
  , HasHeader(..)
    -- * Models
  , Model
  , trainNewModel
  , loadModelFromFile
    -- * Prediction
  , predict
  ) where

import           Numeric.Natural (Natural)

import qualified LightGBM.Internal.CommandLineWrapper as CLW
import qualified LightGBM.Parameters as P

-- N.B.  Right now it's just a data file, but it'll become more
-- (e.g. an hmatrix or some such) as we move forward.
-- | A set of data to use for training or prediction.
data DataSet = DataSet
  { dataPath :: FilePath
  , hasHeader :: HasHeader
  } deriving (Eq, Show)

-- | Describes whether a CSV data file has a header row or not.
newtype HasHeader = HasHeader
  { getHeader :: Bool
  } deriving (Eq, Show)

-- | Load data from a file.
--
-- LightGBM can read data from CSV or TSV files (or from LibSVM
-- formatted files).
--
-- Note that the LightGBM data file format traditionally consists of
-- putting the output (aka the /labels/) in the first column, and the
-- inputs (aka the /features/) in the subsequent columns.  However,
-- you can instruct LightGBM to
--
--    * use some other column for the labels with the 'P.LabelColumn' parameter, and
--    * ignore some of the feature columns with the 'P.IgnoreColumns' parameter.
loadDataFromFile :: HasHeader -> FilePath -> DataSet
loadDataFromFile = flip DataSet

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
  -> DataSet -- ^ Training data
  -> DataSet -- ^ Testing data
  -> Natural -- ^ Number of training rounds
  -> IO (Either CLW.ErrLog Model)
trainNewModel modelOutputPath trainingParams trainingData validationData numRounds = do
  let dataParams = [P.HasHeader (getHeader . hasHeader $ trainingData)]
      runParams =
        [ P.Task P.Train
        , P.TrainingData (dataPath trainingData)
        , P.ValidationData (dataPath validationData)
        , P.Iterations numRounds
        , P.OutputModel modelOutputPath
        ]
  runlog <- CLW.run lightgbmExe $ concat [runParams, trainingParams, dataParams]
  return $ either Left (\_ -> Right $ Model modelOutputPath) runlog

-- | Persisted models can be loaded up and used for prediction.
loadModelFromFile :: FilePath -> Model
loadModelFromFile = Model

-- FIXME:
--  - we might want to return the predictions in a better form
--    than just the file...
--  - Duplication of the exec path between predict and
--    train. Use a Reader monad maybe?
-- | Predict the results of new inputs and persist the results to an
-- output file.
predict ::
     Model -- ^ A model to do prediction with
  -> DataSet -- ^ The new input data for prediction
  -> FilePath -- ^ Where to persist the prediction outputs
  -> IO ()
predict model inputData predictionOutputPath = do
  let dataParams = [P.HasHeader (getHeader . hasHeader $ inputData)]
      runParams =
        [ P.Task P.Predict
        , P.InputModel $ modelPath model
        , P.PredictionData $ dataPath inputData
        , P.OutputResult predictionOutputPath
        ]
  _ <- CLW.run lightgbmExe $ concat [dataParams, runParams]
  return ()
