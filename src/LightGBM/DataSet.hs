-- |

module LightGBM.DataSet (
  -- * Data Handling
    loadDataFromFile
  , DataSet (..)
  , HasHeader(..)
  , toList) where

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

-- FIXME - this is broken for anything with more than one column.
-- This needs to use something like Cassava or Frame more
-- consistently.
-- | Convert a DataSet into a list of records for whatever type is relevant.
toList :: Read a => DataSet -> IO [a]
toList ds =  map read . lines <$> readFile (dataPath ds)
