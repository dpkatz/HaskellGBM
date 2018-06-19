{-# LANGUAGE RecordWildCards #-}

module LightGBM.DataSet (
  -- * Data Handling
    DataSet (..)
  , HasHeader(..)
  , readCsvFile
  , writeCsvFile) where

import           System.Directory (copyFile)

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
readCsvFile :: HasHeader -> FilePath -> DataSet
readCsvFile = flip CSVFile

-- | Write a DataSet out to a CSV file
writeCsvFile ::
     FilePath -- ^ Output path
  -> DataSet -- ^ The data to persist
  -> IO ()
writeCsvFile outPath CSVFile {..} = copyFile dataPath outPath
