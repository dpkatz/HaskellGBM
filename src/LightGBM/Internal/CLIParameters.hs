module LightGBM.Internal.CLIParameters
  ( CommandLineParam(..)
  , TaskType(..)
  ) where

data TaskType
  = Train -- ^ Training
  | Predict -- ^ Prediction
  | ConvertModel -- ^ Conversion into an if-then-else format
  | Refit -- ^ Refitting existing models with new data
  deriving (Eq, Show)

-- | Parameters restricted to the CLI
data CommandLineParam
  = ConfigFile FilePath -- ^ Path to config file
  | Task TaskType -- ^ Task to perform (train, predict, etc.)
  | Header Bool -- ^ True if the input data has a header
