-- | Some utility types used across the library.

{-# LANGUAGE DataKinds #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

-- The no-warn-orphans pragma above protect us from the
-- orphan-instance warning for the Hashable instance of the refined
-- types.

module LightGBM.Utils
  (
    -- * Refined Types
    OneToTwoLeftSemiClosed
  , ProperFraction
  , PositiveInt
    -- * Testing functions
  , fileDiff
  ) where

import           Control.Applicative (liftA2)
import qualified Data.ByteString.Lazy as BSL
import           Data.Function (on)
import           Data.Hashable (Hashable, hashWithSalt)
import qualified Refined as R

-- | A 'Double' in the range [0, 1]
type ProperFraction
   = R.Refined (R.And (R.Not (R.LessThan 0)) (R.Not (R.GreaterThan 1))) Double

-- | A 'Double' in the range [1, 2)
type OneToTwoLeftSemiClosed
   = R.Refined (R.And (R.Not (R.LessThan 1)) (R.LessThan 2)) Double

-- | An 'Int' in the range [1, @'maxBound' :: 'Int'@]
type PositiveInt = R.Refined R.Positive Int

instance (Hashable a, R.Predicate p a) => Hashable (R.Refined p a) where
  hashWithSalt salt refinedA = hashWithSalt salt (R.unrefine refinedA)

-- | Determine if two files are the same (byte identical)
fileDiff :: FilePath -> FilePath -> IO Bool
fileDiff = liftA2 (==) `on` BSL.readFile
