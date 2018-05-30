-- |

module LightGBM.Utils.Test (
  -- * Testing functions
  fileDiff
  ) where

import           Control.Applicative (liftA2)
import qualified Data.ByteString.Lazy as BSL
import           Data.Function (on)

-- | Determine if two files are the same (byte identical)
fileDiff :: FilePath -> FilePath -> IO Bool
fileDiff = liftA2 (==) `on` BSL.readFile
