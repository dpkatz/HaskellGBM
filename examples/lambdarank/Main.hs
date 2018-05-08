-- | LambdaRank example (taken from the LightGBM repo)

{-# LANGUAGE TemplateHaskell #-}

module Main where

import           Refined (refineTH)
import           System.Directory as SD
import           System.FilePath ((</>))

import qualified LightGBM as LGBM
import qualified LightGBM.Parameters as P
import           LightGBM.Utils (fileDiff)

workingDir :: FilePath
workingDir =
  "/Users/dkatz/dev/haskell/hLightGBM/examples/lambdarank"

modelName :: String
modelName = "LightGBM_model.txt"

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
main =
  withCurrentDirectory
    workingDir
    (do let trainingData = loadData (workingDir </> "rank.train")
            testData = loadData (workingDir </> "rank.test")
            modelFile = workingDir </> modelName
            predictionFile = workingDir </> "LightGBM_predict_result.txt"

        model <-
          LGBM.trainNewModel modelFile trainParams trainingData testData 100

        LGBM.predict model testData predictionFile

        modelB <- fileDiff modelName (workingDir </> "golden_model.txt")
        modelP <-
          fileDiff predictionFile (workingDir </> "golden_prediction.txt")
        putStrLn $
          case (modelB, modelP) of
            (True, True) -> "Matched!"
            (False, False) -> "Model and Predictions changed"
            (True, False) -> "Predictions changed"
            (False, True) -> "Model changed"
    )
