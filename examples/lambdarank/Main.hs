-- | LambdaRank example (taken from the LightGBM repo)

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}

module Main where

import           Refined (refineTH)
import           Say (say)
import qualified System.Directory as SD
import           System.FilePath ((</>))

import qualified LightGBM as LGBM
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
loadData :: FilePath -> LGBM.DataSet
loadData = LGBM.loadDataFromFile (LGBM.HasHeader False)

main :: IO ()
main = do
  cwd <- SD.getCurrentDirectory
  SD.withCurrentDirectory
    (cwd </> "examples" </> "lambdarank")
    (do let trainingData = loadData "rank.train"
            testData = loadData "rank.test"
            modelName = "LightGBM_model.txt"
            predictionFile = "LightGBM_predict_result.txt"

        model <-
          LGBM.trainNewModel modelName trainParams trainingData testData 100

        case model of
          Left e -> print e
          Right m -> do
            _ <- LGBM.predict m testData predictionFile
            return ()

        modelB <- fileDiff modelName "golden_model.txt"
        modelP <- fileDiff predictionFile "golden_prediction.txt"
        say $
          case (modelB, modelP) of
            (True, True) -> "Matched!"
            (False, False) -> "Model and Predictions changed"
            (True, False) -> "Predictions changed"
            (False, True) -> "Model changed"
    )
