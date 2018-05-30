-- |

module LightGBM.Utils.Csv (
  -- * Reading from CSV files
  readColumn) where

import qualified Data.ByteString.Lazy as BSL
import           Data.ByteString.Lazy.Char8 (unpack)
import qualified Data.Csv as CSV
import qualified Data.Vector as V

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
    extractColumn n = read . unpack . (V.! n)
