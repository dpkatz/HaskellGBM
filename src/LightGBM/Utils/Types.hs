-- | Some utility types used across the library.

{-# LANGUAGE DataKinds #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

-- The no-warn-orphans pragma above protect us from the
-- orphan-instance warning for the Hashable instance of the refined
-- types.

module LightGBM.Utils.Types
  (
    -- * Refined Types
    OneToTwoLeftSemiClosed
  , ProperFraction
  , LeftOpenProperFraction
  , OpenProperFraction
  , PositiveInt
  , IntGreaterThanOne
  , PositiveDouble
  , NonNegativeDouble

    -- * Logging Types
  , OutLog (..)
  , ErrLog (..)
  ) where

import           Data.Hashable (Hashable, hashWithSalt)
import qualified Data.Text as T
import qualified Refined as R

-- | A 'Double' in the range [0, 1]
type ProperFraction
   = R.Refined (R.And (R.Not (R.LessThan 0)) (R.Not (R.GreaterThan 1))) Double

-- | A 'Double' in the range (0, 1]
type LeftOpenProperFraction
   = R.Refined (R.And (R.GreaterThan 0) (R.Not (R.GreaterThan 1))) Double

-- | A 'Double' in the range (0, 1)
type OpenProperFraction
   = R.Refined (R.And (R.GreaterThan 0) (R.LessThan 1)) Double

-- | A 'Double' in the range [1, 2)
type OneToTwoLeftSemiClosed
   = R.Refined (R.And (R.Not (R.LessThan 1)) (R.LessThan 2)) Double

-- | An 'Int' in the range [1, @'maxBound' :: 'Int'@]
type PositiveInt = R.Refined R.Positive Int

-- | An 'Int' in the range [2, @'maxBound' :: 'Int'@]
type IntGreaterThanOne = R.Refined (R.GreaterThan 1) Int

-- | A 'Double' > 0.0
type PositiveDouble = R.Refined R.Positive Double

-- | A 'Double' >= 0.0
type NonNegativeDouble = R.Refined R.NonNegative Double

instance (Hashable a, R.Predicate p a) => Hashable (R.Refined p a) where
  hashWithSalt salt refinedA = hashWithSalt salt (R.unrefine refinedA)

-- | A transcript of the output logging of LightGBM
newtype OutLog = OutLog T.Text deriving Show

-- | A transcript of the error logging of LightGBM
newtype ErrLog = ErrLog T.Text deriving Show
