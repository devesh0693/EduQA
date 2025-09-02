# backend/apps/qa/document_processor.py
import json
import os
import csv
import numpy as np
from django.conf import settings
from apps.core.models import Document, DocumentEmbedding
from .bert_service import bert_service
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        # Use the exact paths you specified
        self.data_dir = r'D:\Devesh R\P\NLP\educational-qa-system\data\datasets'
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def read_tsv_file(self, file_path, delimiter='\t', max_rows=None):
        """Generic method to read TSV files"""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8', newline='') as file:
                reader = csv.reader(file, delimiter=delimiter)
                headers = next(reader)  # Read header row
                
                for row_num, row in enumerate(reader):
                    if max_rows and row_num >= max_rows:
                        break
                    
                    # Create dictionary from headers and row data
                    if len(row) == len(headers):
                        row_dict = dict(zip(headers, row))
                        data.append(row_dict)
                    else:
                        logger.warning(f"Row {row_num + 1} has {len(row)} columns but expected {len(headers)}")
            
            logger.info(f"Successfully read {len(data)} rows from {file_path}")
            return data, headers
            
        except Exception as e:
            logger.error(f"Error reading TSV file {file_path}: {str(e)}")
            return [], []
    
    def load_squad_data(self, file_name='squad_sample.json'):
        """Load and process SQuAD 2.0 dataset"""
        # Try different possible SQuAD files
        squad_files = [
            'squad_sample.json',
            'educational_qa.json',
            'educational_train.json',
            'educational_val.json'
        ]
        
        data_loaded = None
        for filename in squad_files:
            full_path = os.path.join(self.data_dir, filename)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data_loaded = json.load(f)
                    logger.info(f"Successfully loaded SQuAD data from {filename}")
                    break
                except Exception as e:
                    logger.error(f"Error reading {filename}: {str(e)}")
                    continue
        
        if not data_loaded:
            logger.warning("No SQuAD files found, using sample data")
            return self._create_sample_data()
        
        try:
            # If it's the full SQuAD format
            if 'data' in data_loaded:
                processed_data = []
                for article in data_loaded['data']:
                    title = article.get('title', 'Unknown')
                    for paragraph in article.get('paragraphs', []):
                        context = paragraph.get('context', '')
                        qas = []
                        for qa in paragraph.get('qas', []):
                            question = qa.get('question', '')
                            answers = [ans.get('text', '') for ans in qa.get('answers', [])]
                            if question and answers:
                                qas.append({
                                    'question': question,
                                    'answers': [{'text': ans} for ans in answers]
                                })
                        
                        if qas and context:
                            processed_data.append({
                                'title': f"SQuAD: {title[:50]}...",
                                'content': context,
                                'source': 'squad',
                                'qas': qas
                            })
                return processed_data
            # If it's already in the processed format
            elif isinstance(data_loaded, list):
                return data_loaded
            else:
                logger.error("Unexpected SQuAD data format")
                return self._create_sample_data()
            
        except Exception as e:
            logger.error(f"Error processing SQuAD data: {str(e)}")
            return self._create_sample_data()
    
    def load_ms_marco_data(self, dataset_type='train', max_docs=1000):
        """Load and process MS Marco dataset from JSON and TSV files"""
        processed_data = []
        
        # Try loading from JSON files first
        json_files = [
            f'ms_marco_{dataset_type}.json',
            f'ms_marco_train.json',
            f'ms_marco_val.json',
            'ms_marco_summary.json'
        ]
        
        for json_file in json_files:
            json_path = os.path.join(self.data_dir, json_file)
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list):
                        for item in data[:max_docs//2]:  # Limit to avoid too much data
                            if not isinstance(item, dict):
                                continue
                                
                            query = item.get('query', '')
                            answers = item.get('answers', [])
                            passages = item.get('passages', [])
                            
                            if not query:
                                continue
                                
                            # Use the first passage as content or answers as content
                            content = ''
                            if passages:
                                content = passages[0].get('passage_text', '') if isinstance(passages[0], dict) else str(passages[0])
                            elif answers:
                                content = ' '.join(answers[:3]) if isinstance(answers, list) else str(answers)
                                
                            if content and len(content.strip()) > 20:
                                doc_data = {
                                    'title': f"MS Marco: {query[:50]}...",
                                    'content': content,
                                    'source': 'ms_marco',
                                    'qas': [{
                                        'question': query,
                                        'answers': [{'text': ans} for ans in answers[:2]] if answers else []
                                    }]
                                }
                                processed_data.append(doc_data)
                    
                    logger.info(f"Loaded {len(processed_data)} documents from {json_file}")
                    break  # Use first successful file
                    
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {str(e)}")
                    continue
        
        # Try loading from TSV files if no JSON data
        if not processed_data:
            tsv_files = [
                'ms_marco/queries.train.tsv',
                'ms_marco/queries.dev.tsv',
                'ms_marco/qrels.train.tsv',
                'ms_marco/qrels.dev.tsv'
            ]
            
            for tsv_file in tsv_files:
                tsv_path = os.path.join(self.data_dir, tsv_file)
                if os.path.exists(tsv_path):
                    try:
                        data, headers = self.read_tsv_file(tsv_path, max_rows=max_docs//4)
                        
                        for item in data:
                            # Handle different TSV formats
                            if 'query' in item:
                                query = item.get('query', '')
                                content = item.get('passage', item.get('answer', query))
                            else:
                                # Assume first column is query, second is content
                                values = list(item.values())
                                query = values[0] if values else ''
                                content = values[1] if len(values) > 1 else values[0] if values else ''
                            
                            if query and content and len(content.strip()) > 20:
                                doc_data = {
                                    'title': f"MS Marco TSV: {query[:50]}...",
                                    'content': content,
                                    'source': 'ms_marco_tsv',
                                    'qas': [{
                                        'question': query,
                                        'answers': [{'text': content[:200]}]  # Use beginning of content as answer
                                    }]
                                }
                                processed_data.append(doc_data)
                        
                        logger.info(f"Loaded {len(processed_data)} documents from {tsv_file}")
                        if processed_data:  # Use first successful TSV file
                            break
                            
                    except Exception as e:
                        logger.error(f"Error loading {tsv_file}: {str(e)}")
                        continue
        
        return processed_data if processed_data else self._create_educational_sample_data()
    
    def load_ms_marco_passages(self, max_docs=2000):
        """Load MS Marco passage collection from TSV file"""
        collection_path = os.path.join(self.data_dir, 'ms_marco', 'collection.tsv')
        if not os.path.exists(collection_path):
            logger.warning(f"MS Marco collection file not found at {collection_path}")
            return []
            
        processed_passages = []
        try:
            with open(collection_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f, delimiter='\t')
                for i, row in enumerate(reader):
                    if i >= max_docs:
                        break
                        
                    if len(row) >= 2:
                        doc_id, passage = row[0], ' '.join(row[1:]).strip()
                        if len(passage) > 50:  # Only include passages with reasonable length
                            # Extract a title from the first sentence or first few words
                            first_sentence = passage.split('.')[0]
                            title = first_sentence if len(first_sentence) > 20 else ' '.join(passage.split()[:10])
                            
                            processed_passages.append({
                                'title': f"{title[:100]}..." if len(title) > 100 else title,
                                'content': passage,
                                'source': 'ms_marco',
                                'doc_id': doc_id
                            })
                    elif row:  # Handle case where there's only one column
                        passage = row[0].strip()
                        if len(passage) > 50:
                            first_sentence = passage.split('.')[0]
                            title = first_sentence if len(first_sentence) > 20 else ' '.join(passage.split()[:10])
                            
                            processed_passages.append({
                                'title': f"{title[:100]}..." if len(title) > 100 else title,
                                'content': passage,
                                'source': 'ms_marco',
                                'doc_id': str(i)
                            })
                            
            logger.info(f"Loaded {len(processed_passages)} MS Marco passages")
            return processed_passages
            
        except Exception as e:
            logger.error(f"Error loading MS Marco passages: {str(e)}")
            return []
    
    def get_educational_qa_pairs(self):
        """Return a list of predefined educational QA pairs"""
        qa_pairs = [
            {
                "question": "What is the difference between AI and machine learning?",
                "answer": "AI (Artificial Intelligence) is the broader concept of machines being able to carry out tasks in a way that we would consider 'smart'. Machine Learning is a current application of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. In other words, all machine learning is AI, but not all AI is machine learning.",
                "category": "AI Fundamentals"
            },
            {
                "question": "What are transformers in deep learning?",
                "answer": "Transformers are a type of deep learning model architecture introduced in the paper 'Attention Is All You Need'. They rely entirely on self-attention mechanisms to process sequential data, rather than using recurrent or convolutional layers. Transformers are particularly effective for natural language processing tasks and form the basis of models like BERT, GPT, and T5. Key components include multi-head self-attention, positional encodings, and feed-forward neural networks.",
                "category": "Deep Learning"
            },
            {
                "question": "What are convolutional neural networks used for?",
                "answer": "Convolutional Neural Networks (CNNs) are primarily used for processing structured grid data like images. They are particularly effective for: 1) Image classification and recognition, 2) Object detection in images, 3) Image segmentation, 4) Video analysis, and 5) Medical image analysis. CNNs use convolutional layers that automatically and adaptively learn spatial hierarchies of features through backpropagation.",
                "category": "Computer Vision"
            },
            {
                "question": "What is supervised learning?",
                "answer": "Supervised learning is a type of machine learning where the model is trained on a labeled dataset, which means that each training example is paired with an output label. The model learns to map inputs to outputs based on example input-output pairs. Common supervised learning tasks include classification (predicting categories) and regression (predicting continuous values). Examples include spam detection, image classification, and house price prediction.",
                "category": "Machine Learning"
            },
            {
                "question": "How do neural networks learn?",
                "answer": "Neural networks learn through a process called backpropagation combined with gradient descent. During training: 1) Input data is passed through the network (forward pass), 2) The output is compared to the desired output using a loss function, 3) The error is propagated backward through the network (backward pass), 4) The weights are adjusted to minimize the error. This process repeats over many iterations until the network's performance is satisfactory.",
                "category": "Neural Networks"
            },
            {
                "question": "What is natural language processing?",
                "answer": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. It involves enabling computers to process, understand, generate, and respond to human language in a valuable way. Key applications include machine translation, sentiment analysis, chatbots, speech recognition, and text summarization. Modern NLP heavily relies on deep learning models like transformers.",
                "category": "NLP"
            }
        ]
        
        documents = []
        for i, qa in enumerate(qa_pairs):
            doc = {
                'question': qa['question'],
                'content': qa['answer'],
                'answer': qa['answer'],
                'title': f"{qa.get('category', 'Educational')} QA {i+1}",
                'source': 'educational_qa',
                'dataset': 'educational_qa',
                'is_qa_pair': True,
                'category': qa.get('category', 'General')
            }
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} educational QA pairs")
        return documents
        
    def load_educational_data(self):
        """Load and process educational QA dataset"""
        educational_files = [
            'educational_qa.json',
            'educational_train.json', 
            'educational_val.json'
        ]
        
        processed_data = []
        
        # Add custom QA pairs first
        custom_qa_docs = self.get_educational_qa_pairs()
        for qa_doc in custom_qa_docs:
            doc_data = {
                'title': qa_doc['title'],
                'content': qa_doc['content'],
                'source': 'educational_qa',
                'qas': [{
                    'question': qa_doc['question'],
                    'answers': [{'text': qa_doc['answer']}]
                }],
                'is_qa_pair': True,
                'category': qa_doc.get('category', 'General')
            }
            processed_data.append(doc_data)
        
        # Load additional educational files if they exist
        for file_name in educational_files:
            full_path = os.path.join(self.data_dir, file_name)
            
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Ensure data is in the correct format
                    if not isinstance(data, list):
                        data = [data] if isinstance(data, dict) else []
                    
                    for item in data:
                        if isinstance(item, dict):
                            doc_data = {
                                'title': item.get('title', f'Educational Content - {file_name}'),
                                'content': item.get('context', item.get('content', '')),
                                'source': 'educational',
                                'qas': item.get('qas', []),
                                'is_qa_pair': bool(item.get('question')),
                                'category': item.get('category', 'General')
                            }
                            if doc_data['content'] and len(doc_data['content'].strip()) > 20:
                                processed_data.append(doc_data)
                    
                    logger.info(f"Loaded {len(processed_data)} educational documents from {file_name}")
                    
                except Exception as e:
                    logger.error(f"Error loading educational data from {file_name}: {str(e)}")
                    continue
        
        # If no educational files found, use sample data
        if len(processed_data) <= len(custom_qa_docs):  # Only custom QAs were added
            logger.warning("No additional educational files found, using sample data")
            sample_data = self._create_educational_sample_data()
            if sample_data:
                processed_data.extend(sample_data)
        
        logger.info(f"Total educational documents loaded: {len(processed_data)}")
        return processed_data
    
    def _create_educational_sample_data(self):
        """Create educational sample data with detailed content and QA pairs"""
        return [
            {
                'title': 'Python Programming Fundamentals',
                'content': 'Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in data science, web development, artificial intelligence, and automation. Key features include dynamic typing, automatic memory management, and a large standard library.',
                'source': 'educational',
                'qas': [
                    {
                        'question': 'What are the key features of Python?',
                        'answers': [{
                            'text': 'Key features of Python include: 1) Simple and readable syntax, 2) Dynamic typing, 3) Automatic memory management, 4) Large standard library, 5) Support for multiple programming paradigms, 6) Extensive third-party package ecosystem, and 7) Cross-platform compatibility.'
                        }]
                    },
                    {
                        'question': 'What is Python used for?',
                        'answers': [{
                            'text': 'Python is used for: 1) Web development (Django, Flask), 2) Data science and machine learning (NumPy, pandas, scikit-learn), 3) Automation and scripting, 4) Software development, 5) System administration, 6) Game development, and 7) Desktop applications.'
                        }]
                    }
                ]
            },
            {
                'title': 'Database Systems',
                'content': 'A database is an organized collection of structured information or data, typically stored electronically in a computer system. Database management systems (DBMS) are software systems designed to store, retrieve, define, and manage data in databases. Common types include:
                1. Relational databases (MySQL, PostgreSQL) - Organize data into tables with rows and columns
                2. NoSQL databases (MongoDB, Cassandra) - Designed for unstructured or semi-structured data
                3. Graph databases (Neo4j) - Store data in nodes and relationships
                4. Document stores (MongoDB, CouchDB) - Store data in document format (JSON, BSON)
                5. Key-value stores (Redis, DynamoDB) - Simple data model with key-value pairs',
                'source': 'educational',
                'qas': [
                    {
                        'question': 'What are the main types of databases?',
                        'answers': [{
                            'text': 'The main types of databases are: 1) Relational (SQL) databases like MySQL and PostgreSQL, 2) NoSQL databases like MongoDB and Cassandra, 3) Graph databases like Neo4j, 4) Document stores like MongoDB, and 5) Key-value stores like Redis.'
                        }]
                    },
                    {
                        'question': 'What is the difference between SQL and NoSQL databases?',
                        'answers': [{
                            'text': 'Key differences between SQL and NoSQL databases: 1) SQL databases are table-based while NoSQL can be document, key-value, graph, or wide-column stores. 2) SQL databases have predefined schemas while NoSQL has dynamic schemas. 3) SQL databases are vertically scalable while NoSQL is horizontally scalable. 4) SQL is better for complex queries while NoSQL is better for handling large amounts of data and high load.'
                        }]
                    }
                ]
            },
            {
                'title': 'Machine Learning Fundamentals',
                'content': 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves. The learning process involves: 1) Data collection and preprocessing, 2) Feature selection and engineering, 3) Model training, 4) Model evaluation, and 5) Model deployment. Common types of machine learning include supervised learning, unsupervised learning, and reinforcement learning.',
                'source': 'educational',
                'qas': [
                    {
                        'question': 'What are the main types of machine learning?',
                        'answers': [{
                            'text': 'The three main types of machine learning are: 1) Supervised Learning - learns from labeled training data, 2) Unsupervised Learning - finds patterns in unlabeled data, and 3) Reinforcement Learning - learns by interacting with an environment to achieve goals.'
                        }]
                    },
                    {
                        'question': 'What is the difference between AI and machine learning?',
                        'answers': [{
                            'text': 'AI (Artificial Intelligence) is the broader concept of machines being able to carry out tasks in a smart way. Machine Learning is a current application of AI that provides systems the ability to automatically learn and improve from experience. In other words, all machine learning is AI, but not all AI is machine learning.'
                        }]
                    }
                ]
            },
            {
                'title': 'Web Development Basics',
                'content': 'Web development involves building and maintaining websites and web applications. It can be divided into: 1) Frontend (client-side) development - HTML, CSS, JavaScript, and frameworks like React, Vue, or Angular. 2) Backend (server-side) development - Server, application, and database management using languages like Python, Node.js, Java, or PHP. 3) Full-stack development - Combination of both frontend and backend development. Modern web development also includes concepts like responsive design, progressive web apps (PWAs), and single-page applications (SPAs).',
                'source': 'educational',
                'qas': [
                    {
                        'question': 'What are the main components of web development?',
                        'answers': [{
                            'text': 'The main components of web development are: 1) Frontend (client-side) - HTML, CSS, JavaScript, and frontend frameworks. 2) Backend (server-side) - Server, application, and database. 3) Full-stack - Combination of both frontend and backend development.'
                        }]
                    },
                    {
                        'question': 'What is the difference between frontend and backend development?',
                        'answers': [{
                            'text': 'Frontend development focuses on the user interface and user experience (HTML, CSS, JavaScript), while backend development handles server-side logic, databases, and application programming interfaces (APIs) that power the frontend.'
                        }]
                    }
                ]
            }
        ]
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def get_relevant_chunks(self, question, top_k=5):
        """Get relevant document chunks for a question with enhanced context"""
        try:
            if not question or len(question.strip()) < 3:
                return self._get_fallback_context(question)
                
            question = question.strip().lower()
            
            # Try embeddings first
            try:
                question_embedding = bert_service.get_embeddings(question)
                return self._get_chunks_by_embedding(question, question_embedding, top_k)
            except Exception as e:
                logger.warning(f"Embedding search failed: {str(e)}")
                return self._keyword_based_search(question, top_k)
                
        except Exception as e:
            logger.error(f"Error in get_relevant_chunks: {str(e)}")
            return self._get_fallback_context(question, top_k)
            
    def _get_chunks_by_embedding(self, question, question_embedding, top_k):
        """Get chunks using embedding similarity with enhanced scoring"""
        doc_embeddings = DocumentEmbedding.objects.all().select_related('document')
        if not doc_embeddings:
            logger.warning("No document embeddings found")
            return self._keyword_based_search(question, top_k)
            
        # Pre-process question for keyword matching
        question_terms = set(term.lower() for term in question.split() if len(term) > 2)
        similarities = []
        
        for doc_emb in doc_embeddings:
            try:
                # Convert the stored list back to numpy array
                embedding = np.array(doc_emb.embedding_vector, dtype=np.float32)
                
                # Calculate cosine similarity
                norm_q = np.linalg.norm(question_embedding)
                norm_e = np.linalg.norm(embedding)
                if norm_q == 0 or norm_e == 0:
                    continue
                    
                similarity = np.dot(question_embedding, embedding) / (norm_q * norm_e + 1e-9)
                
                # Get document text for keyword matching
                doc_title = (doc_emb.document.title or '').lower()
                doc_content = (doc_emb.chunk_text or '').lower()
                
                # Boost for keyword matches in title (more important than content)
                title_terms = set(doc_title.split())
                title_match_boost = 0.2 * len(question_terms.intersection(title_terms))
                
                # Smaller boost for content matches
                content_terms = set(doc_content.split())
                content_match_boost = 0.1 * len(question_terms.intersection(content_terms))
                
                # Apply boosts
                similarity += (title_match_boost + content_match_boost)
                
                # Store the enhanced similarity score
                similarities.append((doc_emb, float(similarity)))
                
            except Exception as e:
                logger.error(f"Error processing embedding: {str(e)}")
                
        if not similarities:
            return self._keyword_based_search(question, top_k)
            
        # Sort and get top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        seen_content = set()
        
        for doc_emb, score in similarities:
            if len(results) >= top_k:
                break
                
            content_hash = hash(doc_emb.chunk_text[:500])
            if content_hash in seen_content:
                continue
                
            seen_content.add(content_hash)
            results.append({
                'title': doc_emb.document.title,
                'content': doc_emb.chunk_text,
                'score': score,
                'source': getattr(doc_emb.document, 'source', 'document')
            })
            
        return results if results else self._keyword_based_search(question, top_k)
        
    def _keyword_based_search(self, query, max_chunks=5):
        """Fallback to keyword-based search with improved relevance"""
        try:
            # Extract meaningful query terms (longer than 2 chars, not stopwords)
            query_terms = [term.lower() for term in query.split() 
                         if len(term) > 2 and term.lower() not in {
                             'what', 'where', 'who', 'how', 'when', 'why', 
                             'the', 'and', 'or', 'is', 'are', 'was', 'were'
                         }]
            
            if not query_terms:
                return self._get_fallback_context(query, max_chunks)
                
            documents = Document.objects.filter(is_active=True)
            if not documents.exists():
                return self._get_fallback_context(query, max_chunks)
                
            scored_chunks = []
            query_set = set(query_terms)
            
            for doc in documents:
                # Score based on title matches (higher weight)
                title = (doc.title or '').lower()
                title_terms = set(title.split())
                title_score = 3.0 * len(query_set.intersection(title_terms))
                
                # If title has a good match, boost the score
                if title_score > 0:
                    title_score *= 1.5
                
                # Score content chunks
                chunks = self.chunk_text(doc.content or '')
                for chunk in chunks:
                    chunk_lower = chunk.lower()
                    chunk_terms = set(chunk_lower.split())
                    
                    # Calculate content score with term frequency
                    content_score = 0
                    for term in query_terms:
                        content_score += chunk_lower.count(term) * (2 if len(term) > 4 else 1)
                    
                    # Bonus for exact phrase matches
                    if all(term in chunk_lower for term in query_terms):
                        content_score *= 1.5
                    
                    # Combine scores with title match boost
                    total_score = title_score + content_score
                    
                    if total_score > 0:
                        scored_chunks.append((doc, chunk, total_score))
                        
            scored_chunks.sort(key=lambda x: x[2], reverse=True)
            return [{
                'title': doc.title,
                'content': chunk,
                'score': score / (len(query_terms) * 3) if query_terms else 0.1,
                'source': getattr(doc, 'source', 'document')
            } for doc, chunk, score in scored_chunks[:max_chunks]]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return self._get_fallback_context(query, max_chunks)
            
    def _get_fallback_context(self, query, count=3):
        """Generate fallback context when no good matches are found"""
        return [{
            'title': 'Educational Resource',
            'content': 'I apologize, but I couldn\'t find specific information to answer your question. Please try rephrasing or providing more context.',
            'score': 0.1,
            'source': 'system'
        }]
            
    def process_and_store_documents(self, user):
        """Process datasets and store in database with correct file paths"""
        try:
            all_data = []
            
            # First, load MS Marco passages from collection.tsv (highest quality)
            logger.info("Loading MS Marco passages...")
            ms_marco_passages = self.load_ms_marco_passages(max_docs=2000)
            all_data.extend(ms_marco_passages)
            logger.info(f"Loaded {len(ms_marco_passages)} MS Marco passages")
            
            # Then load SQuAD data (high quality)
            logger.info("Loading SQuAD data...")
            squad_data = self.load_squad_data()
            all_data.extend(squad_data)
            logger.info(f"Loaded {len(squad_data)} SQuAD documents")
            
            # Then load educational data (medium quality)
            logger.info("Loading educational data...")
            edu_data = self.load_educational_data()
            all_data.extend(edu_data)
            logger.info(f"Loaded {len(edu_data)} educational documents")
            
            # Finally, load other MS Marco data (lower quality)
            logger.info("Loading additional MS Marco data...")
            ms_marco_data = self.load_ms_marco_data('train', max_docs=300)
            all_data.extend(ms_marco_data)
            logger.info(f"Loaded {len(ms_marco_data)} additional MS Marco documents")
            
            created_count = 0
            skipped_count = 0
            error_count = 0
            
            logger.info(f"Processing {len(all_data)} total documents")
            
            for doc_data in all_data:
                try:
                    # Skip if no content or invalid content
                    content = doc_data.get('content', '').strip()
                    if not content or len(content) < 10:
                        skipped_count += 1
                        continue
                    
                    # Generate a title if not present
                    title = doc_data.get('title', '')
                    if not title:
                        # Create a title from the first few words of content
                        title = ' '.join(content.split()[:10])
                        if len(title) < 5:  # If still too short, use a default
                            title = 'Untitled Document'
                    
                    # Truncate title to fit in database field
                    title = title[:200]
                    
                    # Check if document with same title and similar content already exists
                    existing = Document.objects.filter(
                        title=title,
                        content__startswith=content[:100]  # Check first 100 chars for similarity
                    ).exists()
                    
                    if existing:
                        skipped_count += 1
                        continue
                    
                    # Create document
                    document = Document.objects.create(
                        title=title,
                        content=content,
                        file_type='json',
                        uploaded_by=user
                    )
                    
                    # Create embeddings for chunks
                    chunks = self.chunk_text(content, chunk_size=400, overlap=50)
                    embedding_count = 0
                    
                    for idx, chunk in enumerate(chunks):
                        try:
                            chunk = chunk.strip()
                            if len(chunk) < 20:  # Skip very short chunks
                                continue
                                
                            embedding = bert_service.get_embeddings(chunk)
                            
                            DocumentEmbedding.objects.create(
                                document=document,
                                chunk_text=chunk[:2000],  # Ensure chunk fits in database field
                                chunk_index=idx,
                                embedding_vector=embedding.tolist()
                            )
                            embedding_count += 1
                            
                        except Exception as e:
                            logger.error(f"Error creating embedding for chunk {idx} in document {document.title}: {str(e)}")
                            continue
                    
                    created_count += 1
                    logger.info(f"Processed document: {document.title} ({embedding_count} embeddings)")
                    
                    # Log progress every 25 documents
                    if created_count % 25 == 0:
                        logger.info(f"Progress: {created_count} created, {skipped_count} skipped, {error_count} errors")
                        
                except Exception as e:
                    doc_title = doc_data.get('title', '')
                    if not doc_title and 'content' in doc_data:
                        # Try to extract a title from content if available
                        content_preview = doc_data['content'][:50].replace('\n', ' ')
                        doc_title = f"Content: {content_preview}..."
                    logger.error(f"Error processing document {doc_title}: {str(e)}")
                    error_count += 1
                    continue
            
            logger.info(f"Document processing completed:")
            logger.info(f"- Created: {created_count} documents")
            logger.info(f"- Skipped: {skipped_count} documents (already existed)")
            logger.info(f"- Errors: {error_count} documents")
            
            return created_count
            
        except Exception as e:
            logger.error(f"Error in process_and_store_documents: {str(e)}")
            return 0
    
    def get_dataset_info(self):
        """Get information about available datasets"""
        info = {
            'squad_files': [],
            'ms_marco_files': [],
            'educational_files': [],
            'ms_marco_dir_files': []
        }
        
        # Check for direct JSON files in datasets directory
        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                if file.startswith('educational_') and file.endswith('.json'):
                    info['educational_files'].append(file)
                elif file.startswith('ms_marco_') and file.endswith('.json'):
                    info['ms_marco_files'].append(file)
                elif 'squad' in file.lower() and file.endswith('.json'):
                    info['squad_files'].append(file)
        
        # Check MS MARCO subdirectory
        ms_marco_dir = os.path.join(self.data_dir, 'ms_marco')
        if os.path.exists(ms_marco_dir):
            info['ms_marco_dir_files'] = [f for f in os.listdir(ms_marco_dir)]
        
        # Add file existence status
        info['paths'] = {
            'data_dir': self.data_dir,
            'ms_marco_dir': ms_marco_dir,
            'data_dir_exists': os.path.exists(self.data_dir),
            'ms_marco_dir_exists': os.path.exists(ms_marco_dir)
        }
        
        return info

# Global instance
document_processor = DocumentProcessor()