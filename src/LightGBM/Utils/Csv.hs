-- |

module LightGBM.Utils.Csv (
  -- * Reading from CSV files
  readColumn
  -- * Reshaping CSV files
  , dropColumns
  , keepColumns
  , dropNamedColumns
  , keepNamedColumns) where

import qualified Data.ByteString.Lazy as BSL
import qualified Data.ByteString.Lazy.Char8 as BSL8
import qualified Data.Csv as CSV
import qualified Data.Foldable as F
import           Data.Maybe (isNothing, catMaybes)
import qualified Data.Vector as V

-- Module-level doctest setup
-- $setup
-- >>> import Data.ByteString.Lazy.Char8 (pack)

-- | Read in the n'th column of a CSV file
readColumn :: Read a => Int -> CSV.HasHeader -> BSL.ByteString -> V.Vector a
readColumn index headerStatus csvData =
  let recs =
        CSV.decode headerStatus csvData :: Either String (V.Vector (V.Vector BSL.ByteString)) in
    case recs of
      Left err -> error err
      Right rows -> V.map (extractColumn index) rows
  where
    extractColumn :: Read c => Int -> V.Vector BSL.ByteString -> c
    extractColumn n = read . BSL8.unpack . (V.! n)


type RawCSV = V.Vector (V.Vector BSL.ByteString)
filterColumns ::
     Foldable t
  => (Int -> t Int -> Bool)
  -> t Int
  -> BSL.ByteString
  -> BSL.ByteString
filterColumns colPred indices csvdata =
  let rawCols = CSV.decode CSV.NoHeader csvdata :: Either String RawCSV
   in case rawCols of
        Left err -> error err
        Right rcs ->
          let newCols = V.map (V.ifilter (\i _ -> i `colPred` indices)) rcs
           in CSV.encode $ V.toList newCols

-- | Filter the columns of a csv file based on an inclusion or
-- exclusion predicate.
filterNamedColumns ::
     (Foldable t, Functor t)
  => (Int -> [Int] -> Bool)
  -> t BSL.ByteString
  -> BSL.ByteString
  -> BSL.ByteString
filterNamedColumns colPred names csvdata =
  let headerLine = head $ BSL8.lines csvdata
      colHeaders = CSV.decode CSV.NoHeader headerLine :: Either String RawCSV
   in case colHeaders of
        Left err -> error err
        Right headerRows ->
          let headers = headerRows V.! 0
              filterIndices = fmap (`V.elemIndex` headers) names
           in case any isNothing filterIndices of
                True -> error "Bad header name!!"
                False ->
                  filterColumns
                    colPred
                    (catMaybes . F.toList $ filterIndices)
                    csvdata

dropNamedColumns ::
     (Foldable t, Functor t)
  => t BSL8.ByteString
  -> BSL8.ByteString
  -> BSL8.ByteString
dropNamedColumns = filterNamedColumns notElem

keepNamedColumns ::
     (Foldable t, Functor t)
  => t BSL8.ByteString
  -> BSL8.ByteString
  -> BSL8.ByteString
keepNamedColumns = filterNamedColumns elem

-- | Drop the selected columns of the CSV file
--
-- >>> csv = pack "h0,h1,h2,h3,h4\n0,1,2,3,4\n10,11,12,13,14\n"
-- >>> dropColumns [1, 3] csv
-- "h0,h2,h4\r\n0,2,4\r\n10,12,14\r\n"
dropColumns :: Foldable t => t Int -> BSL.ByteString -> BSL.ByteString
dropColumns = filterColumns notElem

-- | Keep only the selected columns of the CSV file
--
-- >>> csv = pack "h0,h1,h2,h3,h4\n0,1,2,3,4\n10,11,12,13,14\n"
-- >>> keepColumns [1, 3] csv
-- "h1,h3\r\n1,3\r\n11,13\r\n"
keepColumns :: Foldable t => t Int -> BSL.ByteString -> BSL.ByteString
keepColumns = filterColumns elem
