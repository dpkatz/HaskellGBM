-- | LambdaRank example (taken from the LightGBM repo)

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}

module Main where

import           Refined (refineTH)
import           Say (say)
import qualified System.Directory as SD
import           System.FilePath ((</>))

import qualified LightGBM as LGBM
import qualified LightGBM.DataSet as DS
import qualified LightGBM.Parameters as P
import           LightGBM.Utils.Test (fileDiff)

trainParams :: [P.Param]
trainParams =
  [ P.App P.LambdaRank
  , P.Metric [P.NDCG (Just [1, 3, 5])]
  , P.TrainingMetric True
  , P.LearningRate 0.1
  , P.BaggingFreq $$(refineTH 1)
  , P.BaggingFraction $$(refineTH 0.9)
  , P.MinDataInLeaf 50
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
    (cwd </> "examples" </> "lambdarank")
    (do let trainingData = loadData "rank.train"
            testData = loadData "rank.test"
            predictionFile = "LightGBM_predict_result.txt"

        model <-
          LGBM.trainNewModel trainParams trainingData [testData]

        case model of
          Left e -> print e
          Right m -> do
            LGBM.predict m testData >>= LGBM.writeCsvFile predictionFile

            modelP <- fileDiff predictionFile "golden_prediction.txt"
            say $ if modelP then "Matched!" else "Predictions changed"

            SD.removeFile predictionFile
    )
