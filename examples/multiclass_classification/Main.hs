-- | Multiclass classification

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
  "/Users/dkatz/dev/haskell/hLightGBM/examples/multiclass_classification"

modelName :: String
modelName = "LightGBM_model.txt"

trainParams :: [P.Param]
trainParams =
  [ P.App (P.MultiClass P.MultiClassSimple 5)
  , P.TrainingMetric True
  , P.EarlyStoppingRounds $$(refineTH 10)
  , P.LearningRate 0.05
  ]

-- The data files for this test don't have any headers
loadData :: FilePath -> LGBM.DataSet
loadData = LGBM.loadDataFromFile (LGBM.HasHeader False)

main :: IO ()
main =
  withCurrentDirectory
    workingDir
    (do let trainingData = loadData (workingDir </> "multiclass.train")
            testData = loadData (workingDir </> "multiclass.test")
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
            (False, True) -> "Model changed")
