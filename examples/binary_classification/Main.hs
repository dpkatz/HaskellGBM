-- | Binary classification example (taken from the LightGBM repo)

{-# LANGUAGE TemplateHaskell #-}

module Main where

import           Refined (refineTH)
import qualified System.Directory as SD
import           System.FilePath ((</>))

import qualified LightGBM as LGBM
import qualified LightGBM.Parameters as P
import           LightGBM.Utils (fileDiff)


trainParams :: [P.Param]
trainParams =
  [ P.App P.Binary
  , P.Metric [P.BinaryLogloss, P.AUC]
  , P.TrainingMetric True
  , P.LearningRate 0.1
  , P.NumLeaves 63
  , P.FeatureFraction $$(refineTH 0.8)
  , P.BaggingFreq $$(refineTH 5)
  , P.BaggingFraction $$(refineTH 0.8)
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
    (cwd </> "examples" </> "binary_classification")
    (do let trainingData = loadData "binary.train"
            testData = loadData "binary.test"
            predictionFile = "LightGBM_predict_result.txt"
            modelName = "LightGBM_model.txt"
        model <-
          LGBM.trainNewModel modelName trainParams trainingData testData 100
        LGBM.predict model testData predictionFile
        modelB <- fileDiff modelName "golden_model.txt"
        modelP <- fileDiff predictionFile "golden_prediction.txt"
        putStrLn $
          case (modelB, modelP) of
            (True, True) -> "Matched!"
            (False, False) -> "Model and Predictions changed"
            (True, False) -> "Predictions changed"
            (False, True) -> "Model changed")
