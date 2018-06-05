-- | Titanic survivorship example

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}

module Main where

import qualified Data.ByteString.Lazy as BSL
import qualified Data.Csv as CSV
import qualified Data.Vector as V
import           Refined (refineTH)
import qualified System.Directory as SD
import           System.FilePath ((</>))
import           System.IO (hClose, withFile, IOMode(..))
import qualified System.IO.Temp as TMP

import qualified LightGBM as LGBM
import qualified LightGBM.DataSet as DS
import qualified LightGBM.Parameters as P
import           LightGBM.Utils.Csv (readColumn)

import           ConvertData (csvFilter, predsToKaggleFormat, testFilter)


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
  , P.LabelColumn $ P.ColName "Survived"
  , P.IgnoreColumns [P.ColName "PassengerId"]
  , P.CategoricalFeatures
      [P.ColName "Pclass", P.ColName "Sex", P.ColName "Embarked"]
  ]

loadData :: FilePath -> DS.DataSet
loadData = DS.readCsvFile (DS.HasHeader True)

accuracy :: (Eq a, Fractional f) => [a] -> [a] -> f
accuracy predictions knowns =
  let matches = zipWith (==) predictions knowns
      matchCount = length (filter id matches)
      totalCount = length matches
   in fromIntegral matchCount / fromIntegral totalCount

trainModel :: IO LGBM.Model
trainModel =
  TMP.withSystemTempFile "filtered_train" $ \trainFile trainHandle -> do
    _ <- csvFilter "train_part.csv" trainHandle
    hClose trainHandle
    TMP.withSystemTempFile "filtered_val" $ \valFile valHandle -> do
      _ <- csvFilter "validate_part.csv" valHandle
      hClose valHandle
      let trainingData = loadData trainFile
          validationData = loadData valFile
          predictionFile = "LightGBM_predict_result.txt"
          modelName = "LightGBM_model.txt"
      model <-
        LGBM.trainNewModel modelName trainParams trainingData [validationData]
      case model of
        Left e -> error $ "Error training model:  " ++ show e
        Right m -> do
          print $ "Model trained and saved to file:  " ++ modelName

          predictionSet <- LGBM.predict m validationData
          predictions <- DS.getColumn 0 predictionSet :: IO [Double]
          LGBM.writeCsvFile predictionFile predictionSet

          valData <- BSL.readFile valFile
          let knowns = V.toList $ readColumn 0 CSV.HasHeader valData :: [Int]
          print $ "Self Accuracy:  " ++ show (accuracy (round <$> predictions) knowns :: Double)

          return m

main :: IO ()
main = do
  cwd <- SD.getCurrentDirectory
  SD.withCurrentDirectory
    (cwd </> "examples" </> "titanic")
    (do m <- trainModel
        TMP.withSystemTempFile "filtered_test" $ \testFile testHandle -> do
          _ <- testFilter "test.csv" testHandle
          hClose testHandle
          TMP.withSystemTempFile "predictions" $ \predFile predHandle -> do
            hClose predHandle
            _ <- LGBM.writeCsvFile predFile =<<
                 LGBM.predict m (loadData testFile)
            withFile "TitanicSubmission.csv" WriteMode $ \submHandle -> do
              testBytes <- BSL.readFile testFile
              predBytes <- BSL.readFile predFile
              BSL.hPut submHandle $ predsToKaggleFormat testBytes predBytes)
