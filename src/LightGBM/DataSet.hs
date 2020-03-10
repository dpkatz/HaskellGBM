{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module LightGBM.DataSet (
  -- * Data Handling
    DataSet (..)
  , HasHeader(..)
  , fromCSV
  , fromFrame
  , toCSV
  , toFrame) where

import           Data.ByteString (ByteString)
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Csv as CSV
import qualified Data.Text as T
import qualified Data.Vector as V
import           Data.Vinyl (RecMapMethod, RecordToList, RMap)
import qualified Frames as F
import           Frames.CSV ( ParserOptions(..)
                            , ReadRec
                            , defaultParser
                            , readTable
                            , readTableOpt
                            , writeCSV
                            )
import           Frames.InCore (RecVec)
import           Frames.ShowCSV (ShowCSV)

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
fromCSV :: HasHeader -> FilePath -> DataSet
fromCSV = flip CSVFile

-- | Load data from a 'F.Frame' into a 'DataSet'
--
-- Note that this function causes the creation of a file, and it is up
-- to the caller to control the lifetime of this file.  This function
-- is typically called in a 'Control.Exception.bracket' or a similar
-- facility.  For example:
--
-- > withSystemTempFile "inputFrame" $ \ inputFile inputHandle -> do
-- >   hClose trainHandle
-- >   dataset <- fromFrame inFrame inputFile
--
-- where 'inFrame' is the input 'F.Frame'.
fromFrame ::
     ( F.ColumnHeaders ts
     , RecordToList ts
     , Foldable f
     , RecMapMethod ShowCSV F.ElField ts
     )
  => f (F.Record ts)
  -> FilePath
  -> IO DataSet
fromFrame dframe fname = do
  _ <- writeCSV fname dframe
  return $ fromCSV (HasHeader True) fname

-- | Write a 'DataSet' out to a CSV file.
toCSV ::
     FilePath -- ^ Output path
  -> DataSet -- ^ The data to persist
  -> IO ()
toCSV outPath CSVFile {..} = copyFile dataPath outPath

-- | Convert a 'DataSet' out to a 'F.Frame'.
--
-- If the 'DataSet' doesn't have headers, then 'F.Frame' headers are
-- generated with names 'column_i' where 'i' is the index of the
-- column in question (starting at 0).
--
-- Note that this function is polymorphic in the row type - the caller
-- will have to define that explicitly or in context.  (See the
-- doctest below for a simplistic example.)
--
-- >>> :set -XTypeOperators
-- >>> :set -XDataKinds
-- >>> import Frames ((:->))
-- >>> import qualified Frames as F
-- >>> import System.IO (hPutStrLn, hClose)
-- >>> import System.IO.Temp as TMP
-- >>> :{
--   TMP.withSystemTempFile "toFrameTest" $ \ filepath handle -> do
--     hPutStrLn handle "results\n1\n2\n3\n4\n5"
--     hClose handle
--     let ds = fromCSV (HasHeader True) filepath
--     dsf <- toFrame ds :: IO (F.Frame (F.Record '["results" :-> Int]))
--     return $ length dsf
-- :}
-- 5
--
-- >>> :{
--   TMP.withSystemTempFile "toFrameTest" $ \ filepath handle -> do
--     hPutStrLn handle "1\n2\n3\n4"
--     hClose handle
--     let ds = fromCSV (HasHeader False) filepath
--     dsf <- toFrame ds :: IO (F.Frame (F.Record '["column_0" :-> Int]))
--     return $ length dsf
-- :}
-- 4
toFrame :: (RMap rs, RecVec rs, ReadRec rs) => DataSet -> IO (F.FrameRec rs)
toFrame CSVFile {..} =
  case hasHeader of
    HasHeader True -> F.inCoreAoS $ readTable dataPath
    HasHeader False -> do
      opts <- parseOpts
      F.inCoreAoS $ readTableOpt opts dataPath
  where
    parseOpts :: IO ParserOptions
    parseOpts = do
      colNum <- colCount dataPath
      return
        defaultParser
          { headerOverride =
              Just [T.pack ("column_" ++ show i) | i <- [0 .. (colNum - 1)]]
          }

    colCount :: FilePath -> IO Int
    colCount csvfile = do
      csvdata <- BSL.readFile csvfile
      let foo =
            CSV.decode CSV.NoHeader csvdata :: Either String (V.Vector (V.Vector ByteString))
      case foo of
        Left err ->
          error $
          "Failed to get CSV column count for conversion to Frame:" ++ err
        Right stuff -> return . V.length . V.head $ stuff
