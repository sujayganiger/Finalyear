# --- Standard Library Imports ---
import pickle
import os # Import os to access environment variables
import re
import traceback

# --- Third-party Imports ---
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import streamlit as st # Import streamlit

# --- Colab Userdata Import (Optional, for demonstrating how it's used in the launch cell) ---
# We keep this block, but the API key fetching in load_gemini_pro
# will now use os.environ.get() instead of userdata.get()
# The IS_COLAB flag might still be useful for other Colab-specific logic if added later.
try:
    # Attempt to import userdata if running in Colab
    from google.colab import userdata
    IS_COLAB = True
    print("Running in Google Colab context.")
except ImportError:
    # Define a dummy if not in Colab
    class DummyUserdata:
        def get(self, key):
            # This dummy is less relevant now as app.py reads from env var
            print(f"Warning: Running outside Google Colab. Cannot access userdata secrets directly from app.py.")
            return None
    userdata = DummyUserdata()
    IS_COLAB = False
    print("Could not import google.colab.userdata.")


# --- RAG Data Loading (Modified for Streamlit Caching) ---
@st.cache_resource # Cache this function's result to avoid reloading data on each rerun
def load_rag_data():
    """Loads the RAG data components."""
    try:
        with st.spinner("📂 Loading saved RAG data..."):
            print("📂 Loading saved data...") # Print to console for debugging if needed

            required_files = ['data/article_map.pkl', 'data/act_sections_map.pkl', 'data/act_names.pkl',
                              'data/semantic_chunks.pkl', 'data/faiss_index.bin', 'data/sentence_transformer']
            for f in required_files:
                if not os.path.exists(f):
                    st.error(f"❌ Required data file not found: {f}")
                    st.info("Please ensure you have run the data preparation steps and placed the 'data' folder in the same directory as the app.")
                    # Also print to console for Colab logging
                    print(f"❌ Required data file not found: {f}")
                    return None

            with open('data/article_map.pkl', 'rb') as f:
                article_map = pickle.load(f)

            with open('data/act_sections_map.pkl', 'rb') as f:
                act_sections_map = pickle.load(f)

            with open('data/act_names.pkl', 'rb') as f:
                act_names = pickle.load(f)

            with open('data/semantic_chunks.pkl', 'rb') as f:
                semantic_chunks = pickle.load(f)

            faiss_index = faiss.read_index('data/faiss_index.bin')

            embedder_path = 'data/sentence_transformer'
            if not os.path.exists(embedder_path):
                 st.error(f"❌ Sentence transformer model not found at {embedder_path}")
                 st.info("Please ensure you have run the data preparation steps and placed the 'data' folder in the same directory as the app.")
                 # Console print
                 print(f"❌ Sentence transformer model not found at {embedder_path}")
                 return None
            embedder = SentenceTransformer(embedder_path)

            st.success("✅ RAG Data loaded successfully!")
            # Console print
            print("✅ Data loaded successfully!")
            return {
                'faiss_index': faiss_index,
                'embedder': embedder,
                'semantic_chunks': semantic_chunks,
                'article_map': article_map,
                'act_sections_map': act_sections_map,
                'act_names': act_names
            }

    except Exception as e:
        st.error(f"Error loading RAG data: {str(e)}")
        st.exception(e) # Show traceback in Streamlit
        # Console print
        print(f"Error loading data: {str(e)}")
        traceback.print_exc()
        return None

# --- Language Model Loading (Modified to use Environment Variable) ---
@st.cache_resource # Cache this function's result
def load_gemini_pro():
    """Loads the Gemini language model using an environment variable for API key."""
    with st.spinner("🤖 Loading Gemini language model..."):
        print("🤖 Loading Gemini language model...") # Console print

        # --- Use Environment Variable for API Key ---
        # API key should be passed as an environment variable named 'GOOGLE_API_KEY'
        # This is the standard way to pass config/secrets to processes.
        # When running in Colab using the recommended launch method, the key
        # is fetched from Colab Secrets in the launch cell and passed as an env var.
        api_key = os.environ.get('GOOGLE_API_KEY')

        if not api_key:
            st.error("❌ GOOGLE_API_KEY environment variable not set.")
            st.warning("Please ensure your API key is set as the GOOGLE_API_KEY environment variable when running this app.")
            # Console print
            print("❌ GOOGLE_API_KEY environment variable not set.")
            # If running in Colab and using the standard launch method,
            # this indicates the launch cell didn't correctly pass the key.
            st.info("If running in Google Colab, please check that your 'GOOGLE_API_KEY' secret is set and the launch command correctly passes it as an environment variable.")
            return None
        # --------------------------------------

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')

            # Test connection
            try:
                # Using a minimal API call to verify the key and connection
                model.count_tokens("test connection")
            except Exception as e:
                 st.error(f"Error connecting to Gemini API. Please check your API key and network.")
                 st.warning(f"Details: {e}")
                 # Console print
                 print(f"Error connecting to Gemini API: {e}")
                 return None

            st.success("✅ Gemini 2.0 Flash model loaded successfully!")
            # Console print
            print("✅ Gemini 2.0 Flash model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading Gemini 2.0 Flash model or configuring API: {str(e)}")
            st.exception(e) # Show traceback
            st.warning("Please ensure your `GOOGLE_API_KEY` is correct and valid.")
            # Console print
            print(f"Error loading Gemini 2.0 Flash model or connecting to API: {str(e)}")
            traceback.print_exc()
            return None


# --- Original RAG Helper Functions (Copy-Pasted - No changes needed here) ---
def is_legal_act_query(query, act_names):
    lower_q = query.lower()
    for act_name in act_names:
        escaped_act_name = re.escape(act_name.lower())
        if re.search(r'\b' + escaped_act_name + r'\b', lower_q):
            return True
    section_pattern = r'\bsection\s+(\d+)([a-z]?)'
    if re.search(section_pattern, lower_q):
         if not re.search(r'\barticle\s+\d+[a-z]?.*?\bsection\s+', lower_q):
            return True
    return False

def is_article_specific_query(query):
    lower_q = query.lower()
    article_pattern = r'\barticle\s+(\d+[a-z]?)'
    match = re.search(article_pattern, lower_q)
    if match:
        return True, match.group(1)
    return False, None

def is_direct_text_request(question):
    lower_q = question.lower()
    direct_text_patterns = [
        r'what does article \d+[a-z]? state', r'what does article \d+[a-z]? say', r'what is article \d+[a-z]?\b',
        r'state article \d+[a-z]?', r'what is the title of article \d+[a-z]?', r'text of article \d+[a-z]?',
        r'content of article \d+[a-z]?', r'what does section \d+[a-z]? state', r'what does section \d+[a-z]? say',
        r'what is section \d+[a-z]?\b', r'state section \d+[a-z]?', r'what is the title of section \d+[a-z]?',
        r'text of section \d+[a-z]?', r'content of section \d+[a-z]?'
    ]
    for pattern in direct_text_patterns:
        if re.search(pattern, lower_q):
            return True
    return False

def is_explanation_request(question):
    lower_q = question.lower()
    explanation_patterns = [
        r'explain article \d+[a-z]?', r'explain about article \d+[a-z]?', r'explanation of article \d+[a-z]?',
        r'article \d+[a-z]? explanation', r'article \d+[a-z]? in detail', r'article \d+[a-z]? in brief',
        r'article \d+[a-z]? briefly', r'brief article \d+[a-z]?', r'article \d+[a-z]? deals with',
        r'what does article \d+[a-z]? deal with', r'meaning of article \d+[a-z]?', r'article \d+[a-z]? meaning',
        r'elaborate article \d+[a-z]?', r'elaborate on article \d+[a-z]?', r'completely explain article \d+[a-z]?',
        r'simplify article \d+[a-z]?', r'summary of article \d+[a-z]?', r'summarize article \d+[a-z]?'
    ]
    for pattern in explanation_patterns:
        if re.search(pattern, lower_q):
            return True
    return False

def is_followup_question(question, chat_history):
    """Determine if the current question appears to be a follow-up."""
    if not chat_history:
        return False, None
    lower_q = question.lower()
    followup_patterns = [r'\bit\b', r'\bthis\b', r'\bthose\b', r'\bthat\b', r'\bthe article\b', r'\bthese\b', r'\bthe section\b']
    for pattern in followup_patterns:
        if re.search(pattern, lower_q):
            last_entry = chat_history[-1]
            last_question = last_entry['question']
            is_article, article_num = is_article_specific_query(last_question)
            if is_article:
                return True, article_num
            section_match = re.search(r'\bsection\s+(\d+)([a-z]?)', last_question.lower())
            if section_match:
                section_num = section_match.group(1) + (section_match.group(2) if section_match.group(2) else "")
                return True, f"section {section_num}" # Return section number as reference
    return False, None

# Modified get_context to accept chat_history as an argument
def get_context(question, faiss_index, embedder, semantic_chunks, article_map, act_sections_map, act_names, chat_history):
    lower_q = question.lower()
    context_chunks = []
    retrieved_keys = set()

    # Check if it's a follow-up question and modify the question if needed
    is_followup, reference = is_followup_question(question, chat_history)
    if is_followup and reference:
        print(f"Detected follow-up question referring to: {reference}") # Console print
        if re.match(r'^\d+[a-z]?$', reference): # Article number
            expanded_question = question.replace("it", f"Article {reference}")
            expanded_question = re.sub(r'\bthis\b', f"Article {reference}", expanded_question)
            expanded_question = re.sub(r'\bthe article\b', f"Article {reference}", expanded_question)
            question = expanded_question
            lower_q = question.lower()
            print(f"Expanded question: {expanded_question}") # Console print
        elif "section" in reference.lower(): # Section
             section_match_in_ref = re.search(r'\bsection\s+(\d+)([a-z]?)', reference.lower())
             if section_match_in_ref:
                  section_num_ref = section_match_in_ref.group(1) + (section_match_in_ref.group(2) if section_match_in_ref.group(2) else "")
                  act_name_from_history = None
                  for entry in reversed(chat_history): # Look back in history for Act name
                      # Check if the previous turn referred to this section number
                      if entry.get('section_reference') == section_num_ref:
                          # Then try to find an Act name in that previous question
                          for act_name_check in act_names:
                              if re.search(r'\b' + re.escape(act_name_check.lower()) + r'\b', entry['question'].lower()):
                                  act_name_from_history = act_name_check
                                  break
                          if act_name_from_history: break # Found relevant info, stop looking
                  if act_name_from_history:
                       full_reference = f"{act_name_from_history} section {section_num_ref}"
                       expanded_question = question.replace("it", full_reference)
                       expanded_question = re.sub(r'\bthis\b', full_reference, expanded_question)
                       expanded_question = re.sub(r'\bthe section\b', full_reference, expanded_question)
                       question = expanded_question
                       lower_q = question.lower()
                       print(f"Expanded question with Act from history: {expanded_question}") # Console print
                  else:
                       # If no specific Act found in history for that section, just use the section number reference
                       expanded_question = question.replace("it", reference) # 'reference' is just "section N"
                       expanded_question = re.sub(r'\bthis\b', reference, expanded_question)
                       expanded_question = re.sub(r'\bthe section\b', reference, expanded_question)
                       question = expanded_question
                       lower_q = question.lower()
                       print(f"Expanded question (section, no Act found in history): {expanded_question}") # Console print


    is_article_query, article_num = is_article_specific_query(question)
    if is_article_query and article_num:
        article_key_variants = [f"article {article_num}", f"article {article_num.lower()}", f"Article {article_num}", f"Article {article_num.upper() if article_num.isalpha() else article_num}"]
        for key in article_key_variants:
            if key in article_map:
                print(f"Found direct article match: {key}. Retrieving specific text.") # Console print
                return [article_map[key]] # Direct match found, return only this

    is_legal_act_query_flag = is_legal_act_query(lower_q, act_names)
    if is_legal_act_query_flag:
        section_match = re.search(r'\bsection\s+(\d+)([a-z]?)', lower_q)
        if section_match:
            section_num = section_match.group(1) + (section_match.group(2) if section_match.group(2) else "")
            # Try to find a specific Act mention in the query as well
            act_match_in_query = None
            for act_name in act_names:
                 escaped_act_name = re.escape(act_name.lower())
                 if re.search(r'\b' + escaped_act_name + r'\b', lower_q):
                    act_match_in_query = act_name
                    break # Found the first matching act name

            if act_match_in_query:
                 section_key_variants = [f"{act_match_in_query} section {section_num}", f"{act_match_in_query.lower()} section {section_num.lower()}"] # Add more variations if needed
                 for key in section_key_variants:
                     if key in act_sections_map:
                         print(f"Found direct Act Section match: {key}. Retrieving specific text.") # Console print
                         return [act_sections_map[key]] # Direct match found, return only this

            # If no specific Act + Section match, but a section number was mentioned
            # We could potentially fall through to semantic search, but the original logic
            # seemed to *try* to find a section match without the Act name. Let's keep that logic.
            # Loop through all section keys to find a match based on number alone
            for key, value in act_sections_map.items():
                section_match_in_key = re.search(r'\bsection\s+(\d+)([a-z]?)', key.lower())
                if section_match_in_key:
                    key_section_num = section_match_in_key.group(1) + section_match_in_key.group(2)
                    if key_section_num.lower() == section_num.lower():
                         print(f"Found Section match (by number alone): {key}. Retrieving specific text.") # Console print
                         # Decide whether to return *all* matches or just the first.
                         # Returning the first is simpler and matches the article logic.
                         return [value]
            # If a section number was in the query but *no* section context was found by direct lookup
            print(f"Section number {section_num} detected in query, but no direct Act or Section match found. Proceeding to semantic search.")

        elif is_legal_act_query_flag and not context_chunks: # Legal act name mentioned but no section
            print("Legal act name detected, no specific section found. Proceeding to semantic search.")

    # If no direct Article or Section lookup returned context, perform semantic search
    if not context_chunks:
        print("Using semantic search...") # Console print
        try:
            query_embedding = embedder.encode([question], convert_to_tensor=False) # Ensure numpy array
            faiss.normalize_L2(query_embedding)
            k_semantic = 7
            _, indices = faiss_index.search(np.array(query_embedding).astype('float32'), k=k_semantic)

            # Retrieve unique chunks based on indices
            unique_chunks = set()
            for i in indices[0]:
                if 0 <= i < len(semantic_chunks): # Add bounds check
                     semantic_chunk = semantic_chunks[i]
                     unique_chunks.add(semantic_chunk.strip()) # Use strip() for uniqueness

            context_chunks = list(unique_chunks) # Convert set back to list

        except Exception as e:
            print(f"Error during semantic search: {e}")
            traceback.print_exc()
            # Continue, but context_chunks will be empty or partial


    max_chunks_for_llm = 10 # Limit the number of chunks sent to the LLM
    final_context = context_chunks[:max_chunks_for_llm]

    if not final_context:
         print("Warning: Retrieval returned no context chunks after all attempts.") # Console print

    return final_context

# Modified create_prompt to accept chat_history as an argument
def create_prompt(question, context_list, chat_history):
    """Creates the prompt for the language model."""
    history_context = ""
    # Include a reasonable amount of recent history
    if chat_history and len(chat_history) > 0:
        recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history # Increased history slightly
        history_entries = []
        for entry in recent_history:
            answer_text = str(entry.get('answer', '')) # Ensure answer is string
            # Clean up potential Streamlit markdown like "🔍 **Answer:**\n\n" from previous turns
            # Using re.sub with a non-greedy match and specifying count=1 for efficiency
            cleaned_answer_text = re.sub(r'🔍 \*\*Answer:\*\*(\n\n)?', '', answer_text, count=1)
            cleaned_answer_text = re.sub(r'❌ \*\*Error:\*\*(\n\n)?', '', cleaned_answer_text, count=1) # Also clean up error indicators

            # Limit history length to avoid exceeding context window
            max_hist_len = 500 # characters per turn in history
            hist_q = entry.get('question', 'N/A')
            hist_a = cleaned_answer_text

            if len(hist_q) + len(hist_a) > max_hist_len:
                 # Simple truncation if needed
                 hist_a = hist_a[:max_hist_len - len(hist_q) - 10] + "..." # -10 for "...\n"

            history_entries.append(f"Previous Question: {hist_q}\nPrevious Answer: {hist_a}")

        history_context = "\n\n--- Previous Conversation Context ---\n\n" + "\n\n".join(history_entries) + "\n\n"

    context = "\n\n--- Retrieved Document Chunk ---\n\n".join(context_list)

    # --- Prompting Logic ---
    # Added more specific checks to prioritize direct text/explanation requests
    # based on the *current* question, even if the context isn't a perfect match,
    # as the user intent is clear.

    if is_direct_text_request(question):
        print("Direct text request detected. Using verbatim response prompt.")
        return f"""You are an AI assistant specialized in Indian law. Your primary task is to provide the EXACT text as it appears in the provided context when asked for specific articles or sections.

{history_context}
Carefully read the retrieved text below:
---------------------
{context}
---------------------

Instructions:
1. If the exact text for the requested article or section (e.g., "what does Article 21 state", "text of IPC Section 302") is present in the context, provide ONLY that exact text. Include the official article/section number and title exactly as shown in the text.
2. DO NOT paraphrase, explain, or interpret the content if a direct text request is made and the text is found.
3. If the requested article/section text is NOT present in the context, politely mention that you don't have the specific text and share what information you *do* have from the context that might be related to the user's query or the mentioned article/section number.
4. Prioritize providing the exact text if available and requested.

Question: {question}"""

    elif is_explanation_request(question):
        print("Explanation request detected. Using detailed explanation prompt.")
        return f"""You are an AI assistant specialized in Indian Constitutional law. Your task is to provide a DETAILED AND COMPREHENSIVE EXPLANATION of the relevant part of the law based on the provided context.

{history_context}
Carefully read the retrieved text below:
---------------------
{context}
---------------------

Instructions:
1. Explain the content found in the context related to the user's question in simple, accessible language.
2. Break down complex legal concepts, clauses, terms, and implications from the context.
3. Use formatting (like bullet points, headings, bold text) to make the explanation clear and structured.
4. If specific articles or sections are mentioned in the context and are relevant, explain them. Cite them by number (e.g., Article 21, IPC Section 302).
5. Provide context and practical significance based on the provided text.
6. Aim for a comprehensive explanation, drawing all relevant points from the context.
7. If the requested explanation (e.g., of a specific article) is not possible based *only* on the provided context, politely state that and share what information you *can* explain from the text you received.

Question: {question}"""

    # Default RAG prompt for general questions or if direct text/explanation wasn't requested
    print("Using Standard RAG prompt for general query.")
    return f"""You are an AI assistant providing information about Indian law based on the provided text chunks from the Indian Constitution and other legal acts. Answer the user's question using *only* the information found in the context.

{history_context}
Carefully read the retrieved text below:
---------------------
{context}
---------------------

Based on the text provided, answer the following question: {question}

Instructions:
1. Answer the question clearly and concisely, using information found *only* in the context.
2. Use natural, conversational language rather than overly technical legal jargon unless quoting directly from the text.
3. Use phrases like "According to the text provided...", "Based on Article X...", or "Section Y states..." to refer to the source material in the context.
4. If specific Articles or Sections are mentioned in the context and are relevant to the answer, cite them clearly.
5. If you cannot find a complete answer *within the provided text*, state this politely. Share whatever relevant partial information *is* available in the context (e.g., "The provided text mentions Article 21 but does not directly answer your question about X"). DO NOT make up information or use outside knowledge.
6. Format your answer for readability.

Question: {question}"""
    # --- End Prompting Logic ---


def generate_answer(model, prompt, max_output_tokens=2048):
    """
    Generate answer using the Gemini Flash model via API.
    Args:
        model: The initialized google.generativeai.GenerativeModel object.
        prompt (str): The generated prompt with context.
        max_output_tokens (int): Maximum tokens to generate.
    Returns:
        str: The generated answer text, or an error message.
    """
    try:
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_output_tokens,
            temperature=0.2, # Slightly lower temperature for more focused legal answers
            top_p=0.85,
            top_k=40 # Slightly higher top_k
        )

        # Safety settings - typically more relaxed for RAG on legal text
        # Keep safety settings as they were
        safety_settings = [
            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        if response.candidates:
            if response.candidates[0].content.parts:
                output = response.candidates[0].content.parts[0].text
            else:
                print("Warning: Candidate found but content parts are empty.") # Console print
                print(f"Prompt Feedback: {response.prompt_feedback}") # Console print
                return "Could not generate a response with content."
        else:
            print("Warning: No text generated or response was blocked.") # Console print
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Block reason: {response.prompt_feedback.block_reason}") # Console print
                print(f"Safety Ratings: {response.prompt_feedback.safety_ratings}") # Console print
                # Check if blocking is due to sensitive topic, legal queries can sometimes trigger this
                if response.prompt_feedback.block_reason == genai.enums.BlockedReason.SAFETY:
                     # Offer specific advice for safety blocks on legal topics
                     return "My response was blocked due to safety concerns, which can sometimes happen with legal topics. Please try rephrasing your question more neutrally, avoiding hypothetical scenarios involving harm or illegal activities."
                return f"Response blocked: {response.prompt_feedback.block_reason}"
            else:
                print(f"Prompt Feedback: {response.prompt_feedback}") # Console print
                # More specific message if no candidates/block reason isn't safety
                return "Could not generate a response. The model might have found the query ambiguous or out of scope."


        answer = output.strip()

        return answer

    except Exception as e:
        print(f"Error generating answer with Gemini API: {str(e)}") # Console print
        traceback.print_exc() # Console print
        # Attempt to print details if available
        if 'response' in locals():
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                print(f"Prompt Feedback: {response.prompt_feedback}") # Console print
            if hasattr(response, 'candidates') and not response.candidates:
                print("Response was blocked or had no candidates.") # Console print

        return f"Sorry, I encountered an internal error while generating the answer: {e}"

# --- Streamlit App Structure ---

st.set_page_config(page_title="Indian Law RAG Assistant (Colab)", layout="wide")

st.title("⚖️ Indian Law RAG Assistant (Colab)")
st.info("Ask questions about the Indian Constitution and legal acts based on the provided data.")

# Load RAG data and Language Model using cached functions
# These will run only once per Streamlit session
rag_system_data = load_rag_data()
gemini_model = load_gemini_pro() # This now reads from env var

# Check if setup was successful. If not, display errors and stop the app.
if not rag_system_data or not gemini_model:
    # Display a final critical error message if setup failed
    st.error("🔴 Critical Error: Application setup failed. Please check the console output and the error messages above.")
    st.warning("Ensure all required data files are in the 'data' folder, and that the 'GOOGLE_API_KEY' environment variable is correctly set.")
    st.stop() # Stop the Streamlit app execution

# Extract components for easier access (will always be available if st.stop() wasn't called)
faiss_index = rag_system_data['faiss_index']
embedder = rag_system_data['embedder']
semantic_chunks = rag_system_data['semantic_chunks']
article_map = rag_system_data['article_map']
act_sections_map = rag_system_data['act_sections_map']
act_names = rag_system_data['act_names']

# Initialize chat history in session state if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
# Iterate through history and display messages
for i, chat in enumerate(st.session_state.chat_history):
    # Use Streamlit's built-in chat elements for better UI
    with st.chat_message("user"):
         st.markdown(chat['question'])
    with st.chat_message("assistant"):
         st.markdown(chat['answer'])
    # Optional: Display retrieved context for debugging/transparency
    # with st.expander(f"Context for Turn {i+1}"):
    #      st.json(chat.get('context', 'No context retrieved.')) # Use json for context list

st.markdown("---") # Separator below chat history

# Get user input using st.chat_input (preferred for chat UIs)
# It automatically handles submission on Enter and provides the input string
user_question = st.chat_input("Ask a question about Indian Law...")

# Process question when user_question is submitted via chat_input
if user_question:
     # Add user question to history immediately with a placeholder answer
     # This triggers a rerun, showing the user's message and the "Thinking..." state
     st.session_state.chat_history.append({
         'question': user_question.strip(),
         'answer': "Thinking...", # Placeholder
         'context': [], # Will be filled later
         'act_names': act_names # Store act names relevant to this turn if needed (optional)
     })
     st.rerun() # Rerun to display the user message and spinner

# Logic to process the question if the last entry is the "Thinking..." placeholder
# This runs *after* the rerun triggered by submitting the question
if st.session_state.chat_history and st.session_state.chat_history[-1]['answer'] == "Thinking...":
    question_to_process = st.session_state.chat_history[-1]['question']

    # Use a spinner to indicate work is being done.
    # The spinner will appear below the last displayed chat message because
    # the "Thinking..." message is already part of the history being displayed above.
    # Let's move the spinner into the processing block for better flow.
    # Use a more specific message in the spinner
    with st.spinner(f"Assistant is thinking about '{question_to_process}'..."):
        try:
            # --- Core Logic: Retrieve Context, Create Prompt, Generate Answer ---

            print(f"\nUser Query: {question_to_process}") # Console print
            print("🔎 Retrieving relevant context...") # Console print

            # Pass history to get_context (exclude the current "Thinking..." turn)
            # Need to pass a copy or slice up to the second to last element
            history_for_context = st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 1 else []
            context = get_context(
                question_to_process,
                faiss_index,
                embedder,
                semantic_chunks,
                article_map,
                act_sections_map,
                act_names,
                history_for_context # Pass previous turns
            )

            if not context:
                print("⚠️ No relevant context found for this query.") # Console print
                # Provide helpful suggestions based on available data
                available_articles = []
                # Collect article numbers
                for key in article_map.keys():
                    match = re.search(r'article\s+(\d+[a-z]?)', key.lower())
                    if match:
                        article_num = match.group(1)
                        if article_num not in available_articles:
                            available_articles.append(article_num)
                # Sort numerically for better presentation
                try:
                    available_articles.sort(key=lambda x: int(re.match(r'(\d+)', x).group(1)) if re.match(r'^\d+', x) else x)
                except:
                    # Fallback sort if numeric sort fails
                    available_articles.sort()

                articles_sample = ", ".join(available_articles[:20]) # Display sample articles

                # Collect Act names
                act_names_list = list(act_names) # act_names is a set
                act_names_sample = ", ".join(act_names_list)


                answer = f"🔍 **Answer:**\n\nI apologize, but I don't have enough relevant information in my current knowledge base to answer your question about '{question_to_process.strip()}'. My data primarily covers articles from the Indian Constitution and relevant legal acts.\n\nCould you try asking about a specific article (e.g., Article 21), a legal act (e.g., Indian Penal Code), or a section within an act (e.g., IPC Section 302)?\n\nArticles covered include: {articles_sample}... Legal Acts covered include: {act_names_sample}."

                # Update the last history entry with the final answer and no context
                st.session_state.chat_history[-1]['answer'] = answer
                st.session_state.chat_history[-1]['context'] = [] # No context for this error case

            else: # Context was found
                print(f"✅ Retrieved {len(context)} context chunks.") # Console print
                print("💭 Generating answer...") # Console print

                # Pass history to create_prompt (exclude the current "Thinking..." turn)
                prompt = create_prompt(question_to_process, context, history_for_context) # Pass previous turns

                answer = generate_answer(gemini_model, prompt)
                formatted_answer = f"🔍 **Answer:**\n\n{answer}" # Add Markdown formatting

                # Update the last history entry
                st.session_state.chat_history[-1]['answer'] = formatted_answer
                st.session_state.chat_history[-1]['context'] = context # Store retrieved context for potential inspection

                # Extract referenced article or section from the *original* question for better follow-up handling
                # (The expanded question used for retrieval might be too long or modified)
                is_article, article_num = is_article_specific_query(st.session_state.chat_history[-1]['question'])
                if is_article:
                    st.session_state.chat_history[-1]['article_reference'] = article_num

                section_match = re.search(r'\bsection\s+(\d+)([a-z]?)', st.session_state.chat_history[-1]['question'].lower())
                if section_match:
                    section_num = section_match.group(1) + (section_match.group(2) if section_match.group(2) else "")
                    st.session_state.chat_history[-1]['section_reference'] = section_num


            # Keep chat history to a reasonable size (e.g., last 10 interactions)
            # Pop from the beginning if history exceeds the limit
            while len(st.session_state.chat_history) > 10:
                 st.session_state.chat_history.pop(0)

            # Rerun to update the display with the final answer and remove the spinner
            st.rerun()

        except Exception as e:
            # Handle errors during retrieval or generation
            error_message = f"Sorry, an error occurred during processing: {str(e)}"
            st.error(error_message) # Display error in the UI
            st.exception(e) # Show traceback in the Streamlit UI (useful for debugging)

            # Update the last history entry with the error message
            st.session_state.chat_history[-1]['answer'] = f"❌ **Error:** {error_message}"
            st.session_state.chat_history[-1]['context'] = [] # Clear context on error

            # Rerun to show the error message in the chat history
            st.rerun()


# Add a sidebar with instructions and examples
st.sidebar.title("About")
st.sidebar.info(
    "This is a RAG (Retrieval Augmented Generation) application running in Google Colab, using the Gemini 2.0 Flash model to answer questions about Indian law.\n\n"
    "It retrieves relevant context from a pre-indexed corpus (Constitution Articles, Legal Act Sections) and uses the language model to generate an answer based on the retrieved text.\n\n"
    "To run this in Google Colab:\n"
    "1. Ensure you have the `data` folder containing the processed data files in the same directory as `app.py`.\n"
    "2. Add your Gemini API key as a Colab Secret named `GOOGLE_API_KEY` (🔑 icon on the left).\n"
    "3. Run the Colab cell containing the code to save this script (`%%writefile app.py`).\n"
    "4. Run the Colab cell containing the command to launch the app using `!GOOGLE_API_KEY=\"$GOOGLE_API_KEY\" streamlit run app.py & npx localtunnel --port 8501` (You need to fetch the key from userdata in the cell first, as shown in the example launch code).\n"
    "5. Click the generated `loca.lt` URL."
)

st.sidebar.title("Examples")
st.sidebar.write("- What is the right to equality?")
st.sidebar.write("- What does Article 21 state?")
st.sidebar.write("- Explain Article 21 in detail")
st.sidebar.write("- Would someone qualify for citizenship under Article 6 if they migrated in 1947?")
st.sidebar.write("- What are the sections related to theft in the Indian Penal Code?")
st.sidebar.write("- What does Indian Penal Code Section 302 say?")
st.sidebar.write("- What is covered in the Constitution of India?")
st.sidebar.write("- What does it mean?") # Example follow-up
st.sidebar.write("- What are the fundamental rights?")
