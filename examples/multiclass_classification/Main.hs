-- | Multiclass classification

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
  [ P.App (P.MultiClass P.MultiClassSimple 5)
  , P.TrainingMetric True
  , P.EarlyStoppingRounds $$(refineTH 10)
  , P.LearningRate 0.05
  ]

-- The data files for this test don't have any headers
loadData :: FilePath -> DS.DataSet
loadData = DS.loadDataFromFile (DS.HasHeader False)

main :: IO ()
main = do
  cwd <- SD.getCurrentDirectory
  SD.withCurrentDirectory
    (cwd </> "examples" </> "multiclass_classification")
    (do let trainingData = loadData "multiclass.train"
            testData = loadData "multiclass.test"
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
            (False, True) -> "Model changed")
