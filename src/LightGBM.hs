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
-- >  {-# LANGUAGE TemplateHaskell #-}
-- >
-- >  [...]
-- >
-- >  import           LightGBM ( toCSV
-- >                            , readCsvFile
-- >                            , HasHeader(..)
-- >                            , trainNewModel)
-- >  import qualified LightGBM.Parameters as P
-- >  import           Refined (refineTH)
-- >
-- >  let modelFile = "/path/to/model/output"
-- >      trainingData = readCsvFile (HasHeader False) "/path/to/training/data"
-- >      validationData = readCsvFile (HasHeader False) "/path/to/validation/data"
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
-- >
-- >  modelOut <- trainNewModel trainingParams trainingData validationData
-- >  case modelOut of
-- >      Left err -> ... -- handle the errors
-- >      Right model -> do
-- >          let newData = readCsvFile (HasHeader False) "/path/to/inputs_for_prediction"
-- >              outputFile = "/path/to/prediction_outputs"
-- >          predOut <- predict model [] newData
-- >          case predOut of
-- >              Left err -> ... -- handle the errors
-- >              Right preds -> toCSV outputFile preds
module LightGBM
  ( module LightGBM.DataSet
  , module LightGBM.Model
  ) where

import LightGBM.DataSet
import LightGBM.Model
