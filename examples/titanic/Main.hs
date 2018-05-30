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
import           System.IO (hClose)
import qualified System.IO.Temp as TMP

import qualified LightGBM as LGBM
import qualified LightGBM.Parameters as P
import           LightGBM.Utils.Csv (readColumn)

import           ConvertData (csvFilter)


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
  , P.CategoricalFeatures $
    [P.ColName "Pclass", P.ColName "Sex", P.ColName "Embarked"]
  ]

loadData :: FilePath -> LGBM.DataSet
loadData = LGBM.loadDataFromFile (LGBM.HasHeader True)

accuracy :: (Eq a, Fractional f) => [a] -> [a] -> f
accuracy predictions knowns =
  let matches = zipWith (==) predictions knowns
      matchCount = length (filter id matches)
      totalCount = length matches
   in fromIntegral matchCount / fromIntegral totalCount

main :: IO ()
main = do
  cwd <- SD.getCurrentDirectory
  SD.withCurrentDirectory
    (cwd </> "examples" </> "titanic")
    (TMP.withSystemTempFile "filtered_train" $ \trainFile trainHandle -> do
       hClose trainHandle
       _ <- csvFilter "train_part.csv" trainFile
       TMP.withSystemTempFile "filtered_val" $ \valFile valHandle -> do
         hClose valHandle
         _ <- csvFilter "validate_part.csv" valFile
         let trainingData = loadData trainFile
             testData = loadData valFile
             predictionFile = "LightGBM_predict_result.txt"
             modelName = "LightGBM_model.txt"
         model <-
           LGBM.trainNewModel modelName trainParams trainingData testData 100
         case model of
           Left e -> print e
           Right m -> do
             LGBM.predict m testData predictionFile
             predictions <-
               map read . lines <$> readFile predictionFile :: IO [Double]
             valData <- BSL.readFile valFile
             let knownV = readColumn 0 CSV.HasHeader valData :: V.Vector Int
                 knowns = V.toList knownV
             print $
               "Accuracy:  " ++ show (accuracy (round <$> predictions) knowns :: Double))
