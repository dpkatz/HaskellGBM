-- | Regression example (taken from the LightGBM repo)

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
  "/Users/dkatz/dev/haskell/hLightGBM/examples/regression"

modelName :: String
modelName = "LightGBM_model.txt"

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
loadData :: FilePath -> LGBM.DataSet
loadData = LGBM.loadDataFromFile (LGBM.HasHeader False)

main :: IO ()
main =
  withCurrentDirectory
    workingDir
    (do let trainingData = loadData (workingDir </> "regression.train")
            testData = loadData (workingDir </> "regression.test")
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
