from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY,MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger=get_logger(__name__)

class AnimeRecommendationPipeline:
    def __init__(self,persist_dir="chroma_db"):
        try:

            logger.info("Initializing Anime Recommendation Pipeline")
            self.persist_dir = persist_dir
            vector_builder = VectorStoreBuilder(csv_path="", persist_dir=persist_dir) # jsut initializing the vector store builder

            retriever = vector_builder.load_vector_store().as_retriever() #loading the vector store and creating retriever

            self.recommender = AnimeRecommender(retriever=retriever, api_key=GROQ_API_KEY, model_name=MODEL_NAME)

            logger.info("Anime Recommendation Pipeline initialized successfully")
    
    
        except Exception as e:
            logger.error(f"Failed to initialize Anime Recommendation Pipeline: {str(e)}")
            raise CustomException("Error during Anime Recommendation Pipeline initialization", e)
        
    def recommend(self,query:str) -> str:
        try:
            logger.info(f"Recived a query {query}")
            recommendation = self.recommender.get_recommendation(query)

            logger.info("Recommendation generated sucesfulyy...")
            return recommendation
        except Exception as e:
           
            logger.error(f"Failed to get recommendation {str(e)}")
            raise CustomException("Error during getting recommendation" , e)
            



        
