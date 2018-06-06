-- | Regression example (taken from the LightGBM repo)

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
  [ P.App $ P.Regression P.L2
  , P.TrainingMetric True
  , P.LearningRate 0.05
  , P.FeatureFraction $$(refineTH 0.9)
  , P.BaggingFreq $$(refineTH 5)
  , P.BaggingFraction $$(refineTH 0.8)
  , P.MinDataInLeaf 100
  , P.MinSumHessianInLeaf 5.0
  , P.IsSparse True
  ]

-- The data files for this test don't have any headers
loadData :: FilePath -> DS.DataSet
loadData = DS.readCsvFile (DS.HasHeader False)

main :: IO ()
main = do
  cwd <- SD.getCurrentDirectory
  SD.withCurrentDirectory
    (cwd </> "examples" </> "regression")
    (do let trainingData = loadData "regression.train"
            testData = loadData "regression.test"
            predictionFile = "LightGBM_predict_result.txt"

        model <-
          LGBM.trainNewModel trainParams trainingData [testData]
        case model of
          Left e -> sayErrShow e
          Right m -> do
            predResults <- LGBM.predict m [] testData
            case predResults of
              Left e -> sayErrShow e
              Right preds -> LGBM.writeCsvFile predictionFile preds

            modelP <- fileDiff predictionFile "golden_prediction.txt"
            say $ if modelP then "Matched!" else "Predictions changed"

            SD.removeFile predictionFile
    )
