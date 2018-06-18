-- | Binary classification example (taken from the LightGBM repo)

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}

module Main where

import           Refined (refineTH)
import           Say (say, sayErrShow)
import qualified System.Directory as SD
import           System.FilePath ((</>))

import qualified LightGBM as LGBM
import qualified LightGBM.DataSet as DS
import qualified LightGBM.Parameters as P
import           LightGBM.Utils.Test (fileDiff)

trainParams :: [P.Param]
trainParams =
  [ P.Objective $ P.BinaryClassification []
  , P.Metric [P.BinaryLogloss, P.AUC]
  , P.TrainingMetric True
  , P.LearningRate $$(refineTH 0.1)
  , P.NumLeaves $$(refineTH 63)
  , P.FeatureFraction $$(refineTH 0.8)
  , P.BaggingFreq $$(refineTH 5)
  , P.BaggingFraction $$(refineTH 0.8)
  , P.MinDataInLeaf 50
  , P.MinSumHessianInLeaf $$(refineTH 5.0)
  , P.IsSparse True
  ]

-- The data files for this test don't have any headers
loadData :: FilePath -> DS.DataSet
loadData = DS.fromCSV (DS.HasHeader False)

main :: IO ()
main = do
  cwd <- SD.getCurrentDirectory
  SD.withCurrentDirectory
    (cwd </> "examples" </> "binary_classification")
    (do let trainingData = loadData "binary.train"
            testData = loadData "binary.test"
            predictionFile = "LightGBM_predict_result.txt"
        model <-
          LGBM.trainNewModel trainParams trainingData [testData]
        case model of
          Left e -> sayErrShow e
          Right m -> do
            predResults <- LGBM.predict m [] testData
            case predResults of
              Left e -> sayErrShow e
              Right preds -> LGBM.toCSV predictionFile preds

            modelP <- fileDiff predictionFile "golden_prediction.txt"
            say $ if modelP then "Matched!" else "Predictions changed"

            SD.removeFile predictionFile
    )
