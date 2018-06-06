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
loadData = DS.readCsvFile (DS.HasHeader False)

main :: IO ()
main = do
  cwd <- SD.getCurrentDirectory
  SD.withCurrentDirectory
    (cwd </> "examples" </> "multiclass_classification")
    (do let trainingData = loadData "multiclass.train"
            testData = loadData "multiclass.test"
            predictionFile = "LightGBM_predict_result.txt"

        model <-
          LGBM.trainNewModel trainParams trainingData [testData]

        case model of
          Left e -> print e
          Right m -> do
            LGBM.predict m [] testData >>= LGBM.writeCsvFile predictionFile

            modelP <- fileDiff predictionFile "golden_prediction.txt"
            say $ if modelP then "Matched!" else "Predictions changed"

            SD.removeFile predictionFile
    )
