-- |

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module ConvertData (csvFilter) where

import qualified Data.Foldable as Foldable
import           Control.Exception (IOException)
import qualified Control.Exception as Exception
import qualified Data.ByteString.Lazy as BSL
import           Data.Csv ((.:), (.=))
import qualified Data.Csv as CSV
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Vector as V

-- Parsing the Titanic data
data Sex
  = Male
  | Female
  deriving (Eq, Show)

data Embarkation
  = Cherbourg
  | Queenstown
  | Southampton
  | Other T.Text
  deriving (Eq, Show)

newtype Survived =
  Survived Bool
  deriving (Eq, Show, Ord)

data PassengerRecord = PR
  { passengerID :: Int
  , survived :: Survived
  , passengerClass :: Int
  , name :: String
  , sex :: Sex
  , age :: Maybe Double
  , siblingsAndSpouses :: Int
  , parentsAndChildren :: Int
  , ticket :: String
  , fare :: Double
  , cabin :: Maybe String
  , embarkationPoint :: Embarkation
  } deriving (Eq, Show)

instance CSV.FromField Survived where
  parseField "0" = pure $ Survived False
  parseField _ = pure $ Survived True

instance CSV.ToField Survived where
  toField (Survived True) = "1"
  toField (Survived False) = "0"

instance CSV.FromField Sex where
  parseField val =
    let lower = T.toLower . TE.decodeUtf8 $ val
     in case lower of
          "female" -> pure Female
          "male" -> pure Male
          other -> error ("Incorrect sex detected:  " ++ show other)

instance CSV.ToField Sex where
  toField Male = "1"
  toField Female = "0"

instance CSV.FromField Embarkation where
  parseField "S" = pure Southampton
  parseField "C" = pure Cherbourg
  parseField "Q" = pure Queenstown
  parseField other = pure $ Other (TE.decodeUtf8 other)

instance CSV.ToField Embarkation where
  toField Cherbourg = "0"
  toField Queenstown = "1"
  toField Southampton = "2"
  toField (Other _) = "3"

instance CSV.FromNamedRecord PassengerRecord where
  parseNamedRecord r =
    PR
    <$> r .: "PassengerId"
    <*> r .: "Survived"
    <*> r .: "Pclass"
    <*> r .: "Name"
    <*> r .: "Sex"
    <*> r .: "Age"
    <*> r .: "SibSp"
    <*> r .: "Parch"
    <*> r .: "Ticket"
    <*> r .: "Fare"
    <*> r .: "Cabin"
    <*> r .: "Embarked"

-- | When writing the output CSV, we want to strip out irrelevant
-- features (like name and passengerID and cabin) and convert the
-- categorical features into integral values (which is what LightGBM
-- requires at the moment).
instance CSV.ToNamedRecord PassengerRecord where
  toNamedRecord PR {..} =
    CSV.namedRecord
      [ "Survived" .= survived
      , "Pclass" .= passengerClass
      , "Sex" .= sex
      , "Age" .= age
      , "SibSp" .= siblingsAndSpouses
      , "Parch" .= parentsAndChildren
      , "Fare" .= fare
      , "Embarked" .= embarkationPoint
      ]

instance CSV.DefaultOrdered PassengerRecord where
  headerOrder _ =
    CSV.header
      ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

encodeItems :: V.Vector PassengerRecord -> BSL.ByteString
encodeItems = CSV.encodeDefaultOrderedByName . Foldable.toList

encodeFilteredRecordsToFile ::
     FilePath -> V.Vector PassengerRecord -> IO (Either String ())
encodeFilteredRecordsToFile filePath =
  catchShowIO . BSL.writeFile filePath . encodeItems

catchShowIO :: IO a -> IO (Either String a)
catchShowIO action =
  fmap Right action
    `Exception.catch` handleIOException
  where
    handleIOException
      :: IOException
      -> IO (Either String a)
    handleIOException =
      return . Left . show

importNamedCSV :: CSV.FromNamedRecord a => FilePath -> IO [a]
importNamedCSV filename = do
  bytes <- BSL.readFile filename
  case CSV.decodeByName bytes of
    Left err -> do putStrLn err
                   return []
    Right drv -> return . V.toList . snd $ drv

csvFilter :: FilePath -> FilePath -> IO (Either String ())
csvFilter initial final = do
  prs <- V.fromList <$> importNamedCSV initial
  encodeFilteredRecordsToFile final prs
