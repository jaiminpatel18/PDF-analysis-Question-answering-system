import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from keybert import KeyBERT

class DocumentAnalyzer:
    def __init__(self):
        # Initialize KeyBERT for keyword extraction
        # Note: This will download a model on first run
        self.keyword_model = KeyBERT()
    
    def generate_summary(self, text, max_length=500):
        """
        Generate a summary of the text using extractive summarization
        """
        # Simple extractive summarization using sentence importance
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        
        # Remove very short sentences and clean
        sentences = [s.strip() for s in sentences if len(s.strip()) > 40]
        
        if not sentences:
            return "Could not generate summary: text too short or improperly formatted."
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            # Generate TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores based on TF-IDF values
            sentence_scores = np.array([np.mean(tfidf_matrix[i].toarray()) for i in range(len(sentences))])
            
            # Get top sentences based on score
            top_indices = sentence_scores.argsort()[-5:][::-1]  # Top 5 sentences
            
            # Sort indices by their position in the original text
            top_indices = sorted(top_indices)
            
            # Build summary from top sentences
            summary = " ".join([sentences[i] for i in top_indices])
            
            # Truncate if needed
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
                
            return summary
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def extract_key_concepts(self, text, num_concepts=5):
        """
        Extract key concepts/keywords from the text
        """
        try:
            # Extract keywords using KeyBERT
            keywords = self.keyword_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2),  # Allow single words and bigrams
                stop_words='english', 
                use_maxsum=True, 
                top_n=num_concepts
            )
            
            # Format the results
            return [{"keyword": kw, "score": round(score, 2)} for kw, score in keywords]
        
        except Exception as e:
            return [{"keyword": "Error extracting keywords", "score": 0}]
    
    def detect_topics(self, text, num_topics=3):
        """
        Detect main topics in the text
        """
        # Split text into paragraphs for topic detection
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 100]
        
        if len(paragraphs) < 3:
            # Not enough content for meaningful topic clustering
            return ["Not enough content for topic detection"]
        
        try:
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            
            # Generate TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(paragraphs)
            
            # Apply K-means clustering
            km = KMeans(n_clusters=min(num_topics, len(paragraphs)), random_state=42)
            km.fit(tfidf_matrix)
            
            # Get top terms per cluster
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names_out()
            
            topics = []
            for i in range(min(num_topics, len(km.cluster_centers_))):
                # Get top 5 terms for each topic
                top_terms = [terms[ind] for ind in order_centroids[i, :5]]
                topics.append(f"Topic {i+1}: {', '.join(top_terms)}")
            
            return topics
            
        except Exception as e:
            return [f"Error detecting topics: {str(e)}"]