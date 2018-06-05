{-# LANGUAGE RecordWildCards #-}

module LightGBM.DataSet (
  -- * Data Handling
    DataSet (..)
  , HasHeader(..)
  , loadDataFromFile
  , writeDataToFile
  , getColumn) where

import qualified Data.ByteString.Lazy as BSL
import qualified Data.Csv as CSV
import qualified Data.Vector as V
import           System.Directory (renameFile)

import           LightGBM.Utils.Csv (readColumn)

-- N.B.  Right now it's just a data file, but we can add better types
-- (e.g. some sort of dataframe) as other options as we move forward.
-- | A set of data to use for training or prediction.
--
--
data DataSet = CSVFile
  { dataPath :: FilePath
  , hasHeader :: HasHeader
  } deriving (Eq, Show)

-- | Describes whether a CSV data file has a header row or not.
newtype HasHeader = HasHeader
  { getHeader :: Bool
  } deriving (Eq, Show)

-- | Load data from a file.
--
-- LightGBM can read data from CSV or TSV files (or from LibSVM
-- formatted files).
--
-- Note that the LightGBM data file format traditionally consists of
-- putting the output (aka the /labels/) in the first column, and the
-- inputs (aka the /features/) in the subsequent columns.  However,
-- you can instruct LightGBM to
--
--    * use some other column for the labels with the 'P.LabelColumn' parameter, and
--    * ignore some of the feature columns with the 'P.IgnoreColumns' parameter.
loadDataFromFile :: HasHeader -> FilePath -> DataSet
loadDataFromFile = flip CSVFile

-- | Write a DataSet out to a CSV file
writeDataToFile :: FilePath     -- ^ Output path
                -> DataSet      -- ^ The data to persist
                -> IO ()
writeDataToFile outPath CSVFile {..} = renameFile dataPath outPath

-- | Convert a DataSet into a list of records for whatever type is relevant.
getColumn :: Read a => Int -> DataSet -> IO [a]
getColumn colIndex CSVFile {..} =
  V.toList . readColumn colIndex (conv hasHeader) <$> BSL.readFile dataPath
  where
    conv (HasHeader True) = CSV.HasHeader
    conv (HasHeader False) = CSV.NoHeader
