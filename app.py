# --- Standard Library Imports ---
import pickle
import os # Import os to access environment variables
import re
import traceback

# --- Third-party Imports ---
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer # Keep this import
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
    """Loads the RAG data components and the Sentence Transformer model from Hugging Face."""
    try:
        with st.spinner("📂 Loading saved RAG data and embedding model..."):
            print("📂 Loading saved data and embedding model...") # Print to console for debugging if needed

            # List of required data files (excluding the local model directory)
            required_data_files = ['data/article_map.pkl', 'data/act_sections_map.pkl', 'data/act_names.pkl',
                                   'data/semantic_chunks.pkl', 'data/faiss_index.bin']

            for f in required_data_files:
                if not os.path.exists(f):
                    st.error(f"❌ Required data file not found: {f}")
                    st.info("Please ensure you have run the data preparation steps and placed the 'data' folder in the same directory as the app.")
                    # Also print to console for Colab logging
                    print(f"❌ Required data file not found: {f}")
                    return None

            # --- Load data files ---
            with open('data/article_map.pkl', 'rb') as f:
                article_map = pickle.load(f)
            print("✓ Loaded data/article_map.pkl")

            with open('data/act_sections_map.pkl', 'rb') as f:
                act_sections_map = pickle.load(f)
            print("✓ Loaded data/act_sections_map.pkl")

            with open('data/act_names.pkl', 'rb') as f:
                act_names = pickle.load(f)
            print("✓ Loaded data/act_names.pkl")

            with open('data/semantic_chunks.pkl', 'rb') as f:
                semantic_chunks = pickle.load(f)
            print("✓ Loaded data/semantic_chunks.pkl")

            faiss_index = faiss.read_index('data/faiss_index.bin')
            print("✓ Loaded data/faiss_index.bin")

            # --- Load SentenceTransformer model directly from Hugging Face ---
            # IMPORTANT: Use the SAME model name that was used to build the FAISS index
            model_name = 'paraphrase-multilingual-mpnet-base-v2'
            print(f"Loading Sentence Transformer model '{model_name}' from Hugging Face...")
            try:
                 embedder = SentenceTransformer(model_name) # Loading from Hugging Face
                 print(f"✅ Sentence transformer model '{model_name}' loaded successfully from Hugging Face!")
            except Exception as e:
                 st.error(f"❌ Error loading Sentence Transformer model '{model_name}' from Hugging Face: {str(e)}")
                 st.info("Please check your internet connection or ensure the model name is correct.")
                 # Console print
                 print(f"❌ Error loading Sentence Transformer model '{model_name}': {str(e)}")
                 traceback.print_exc()
                 return None
            # --- End Loading from Hugging Face ---

            st.success("✅ RAG Data and Embedding Model loaded successfully!")
            # Console print
            print("✅ Data and Model loaded successfully!")
            return {
                'faiss_index': faiss_index,
                'embedder': embedder, # Pass the loaded embedder
                'semantic_chunks': semantic_chunks,
                'article_map': article_map,
                'act_sections_map': act_sections_map,
                'act_names': act_names
            }

    except Exception as e:
        st.error(f"Error loading RAG data or embedding model: {str(e)}")
        st.exception(e) # Show traceback in Streamlit
        # Console print
        print(f"Error loading data or embedding model: {str(e)}")
        traceback.print_exc()
        return None

# --- Language Model Loading (Modified to use Environment Variable) ---
@st.cache_resource # Cache this function's result
def load_gemini_pro():
    """Loads the Gemini language model using an environment variable for API key."""
    with st.spinner("🤖 Loading Gemini language model..."):
        print("🤖 Loading Gemini language model...") # Console print

        # --- Use Environment Variable for API Key ---
        api_key = os.environ.get('GOOGLE_API_KEY')

        if not api_key:
            st.error("❌ GOOGLE_API_KEY environment variable not set.")
            st.warning("Please ensure your API key is set as the GOOGLE_API_KEY environment variable when running this app.")
            print("❌ GOOGLE_API_KEY environment variable not set.")
            st.info("If running in Google Colab, please check that your 'GOOGLE_API_KEY' secret is set and the launch command correctly passes it as an environment variable.")
            return None
        # --------------------------------------

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')

            # Test connection
            try:
                model.count_tokens("test connection")
            except Exception as e:
                 st.error(f"Error connecting to Gemini API. Please check your API key and network.")
                 st.warning(f"Details: {e}")
                 print(f"Error connecting to Gemini API: {e}")
                 return None

            st.success("✅ Gemini 2.0 Flash model loaded successfully!")
            print("✅ Gemini 2.0 Flash model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading Gemini 2.0 Flash model or configuring API: {str(e)}")
            st.exception(e) # Show traceback
            st.warning("Please ensure your `GOOGLE_API_KEY` is correct and valid.")
            print(f"Error loading Gemini 2.0 Flash model or connecting to API: {str(e)}")
            traceback.print_exc()
            return None


# --- Original RAG Helper Functions (Copy-Pasted - No changes needed here) ---
# These functions were already provided and should work with the new loading method
def is_legal_act_query(query, act_names):
    lower_q = query.lower()
    for act_name in act_names:
        # Use case-insensitive matching for act names from the list
        # Also handle shortened names used in mapping
        cleaned_act_name = re.sub(r',\s*\d{4}$', '', act_name).strip().lower()
        escaped_act_name = re.escape(cleaned_act_name)
        # Look for the act name as a whole word
        if re.search(r'\b' + escaped_act_name + r'\b', lower_q):
            return True
    # Also check for section number format common in acts if no specific act name was found
    section_pattern = r'\bsection\s+(\d+)([a-z]?)'
    if re.search(section_pattern, lower_q):
         # Ensure it's not part of an Article reference
         if not re.search(r'\barticle\s+\d+[a-z]?.*?\bsection\s+', lower_q):
            return True
    return False


def is_article_specific_query(query):
    lower_q = query.lower()
    # Pattern to match "article" followed by digits and optional letter (case insensitive)
    article_pattern = r'\barticle\s+(\d+[a-z]?)\b'
    match = re.search(article_pattern, lower_q)
    if match:
        return True, match.group(1)
    return False, None

def is_direct_text_request(question):
    lower_q = question.lower()
    # Refined patterns to be more precise and handle both article and section requests
    direct_text_patterns = [
        r'what does (?:article|section) \d+[a-z]? state',
        r'what does (?:article|section) \d+[a-z]? say',
        r'what is (?:article|section) \d+[a-z]?\b',
        r'state (?:article|section) \d+[a-z]?',
        r'text of (?:article|section) \d+[a-z]?',
        r'content of (?:article|section) \d+[a-z]?',
        r'provide the text of (?:article|section) \d+[a-z]?'
    ]
    # Add patterns for asking for title specifically
    title_patterns = [
        r'what is the title of (?:article|section) \d+[a-z]?',
        r'title of (?:article|section) \d+[a-z]?'
    ]

    for pattern in direct_text_patterns + title_patterns:
        if re.search(pattern, lower_q):
            return True
    return False

def is_explanation_request(question):
    lower_q = question.lower()
    # Refined patterns for explanation requests
    explanation_patterns = [
        r'explain (?:article|section) \d+[a-z]?',
        r'explain about (?:article|section) \d+[a-z]?',
        r'explanation of (?:article|section) \d+[a-z]?',
        r'(?:article|section) \d+[a-z]? explanation',
        r'(?:article|section) \d+[a-z]? in detail',
        r'(?:article|section) \d+[a-z]? in brief',
        r'(?:article|section) \d+[a-z]? briefly',
        r'brief (?:article|section) \d+[a-z]?',
        r'(?:article|section) \d+[a-z]? deals with',
        r'what does (?:article|section) \d+[a-z]? deal with',
        r'meaning of (?:article|section) \d+[a-z]?',
        r'(?:article|section) \d+[a-z]? meaning',
        r'elaborate (?:on )?(?:article|section) \d+[a-z]?',
        r'completely explain (?:article|section) \d+[a-z]?',
        r'simplify (?:article|section) \d+[a-z]?',
        r'summar(?:y|ize) (?:article|section) \d+[a-z]?'
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
    # Check for common pronoun/reference words
    followup_patterns = [r'\bit\b', r'\bthis\b', r'\bthose\b', r'\bthat\b', r'\bthese\b', r'\bthe article\b', r'\bthe section\b']

    # Check if any followup patterns are in the question AND if the last turn contained a reference
    last_entry = chat_history[-1]
    last_article_ref = last_entry.get('article_reference')
    last_section_ref = last_entry.get('section_reference')
    last_act_name_ref = last_entry.get('act_name_reference') # Might need to store this in the processing loop if possible

    is_pronoun_followup = any(re.search(pattern, lower_q) for pattern in followup_patterns)

    if is_pronoun_followup:
        if last_article_ref:
            # Return the article number from the previous turn
            print(f"Detected pronoun-based follow-up, previous turn referenced Article {last_article_ref}")
            return True, f"Article {last_article_ref}"
        elif last_section_ref:
             # Try to find the Act name from the previous turn's question if available
             act_name_from_prev_q = None
             prev_question = last_entry.get('question', '').lower()
             # Look for an act name in the previous question
             # Use the act_names list from the loaded data
             # (Assume act_names is accessible globally or passed - let's ensure it's passed or accessible)
             # For simplicity, let's pass act_names to this function
             # (Wait, act_names is in rag_system_data, which is global after load. Access it directly)
             global rag_system_data # Declare global usage
             if rag_system_data and rag_system_data.get('act_names'):
                 for act_name_check in rag_system_data['act_names']:
                     # Use the same cleaned name logic as is_legal_act_query
                     cleaned_act_name_check = re.sub(r',\s*\d{4}$', '', act_name_check).strip().lower()
                     if re.search(r'\b' + re.escape(cleaned_act_name_check) + r'\b', prev_question):
                         act_name_from_prev_q = act_name_check
                         break

             if act_name_from_prev_q:
                  print(f"Detected pronoun-based follow-up, previous turn referenced {act_name_from_prev_q} Section {last_section_ref}")
                  return True, f"{act_name_from_prev_q} Section {last_section_ref}"
             else:
                  print(f"Detected pronoun-based follow-up, previous turn referenced Section {last_section_ref} (Act name not found in previous turn)")
                  return True, f"Section {last_section_ref}" # Just return section if Act name isn't clear

    return False, None # Not a recognized follow-up or no relevant reference in history


# Modified get_context to accept chat_history as an argument
def get_context(question, faiss_index, embedder, semantic_chunks, article_map, act_sections_map, act_names, chat_history):
    lower_q = question.lower()
    context_chunks = []
    retrieved_keys = set() # Keep track of keys already retrieved directly

    # Check if it's a follow-up question and modify the question if needed
    # The follow-up logic now returns the reference (e.g., "Article 21" or "IPC Section 302")
    is_followup, reference_text = is_followup_question(question, chat_history)

    # --- Direct Lookup Logic ---
    # Prioritize direct lookup for specific article or section references,
    # including references identified in follow-up questions.

    # Check if the *current* question or the *follow-up reference* specifies an article number
    is_article_query_curr, article_num_curr = is_article_specific_query(question)
    is_article_query_followup = False
    article_num_followup = None

    if reference_text:
        is_article_query_followup, article_num_followup = is_article_specific_query(reference_text)

    # If an article number is found either in the current query or the follow-up reference
    if (is_article_query_curr and article_num_curr) or (is_article_query_followup and article_num_followup):
         article_num_to_lookup = article_num_curr if is_article_query_curr else article_num_followup
         print(f"Attempting direct lookup for Article: {article_num_to_lookup}")
         # Try lookup with various casing/spacing options
         article_key_variants = [f"article {article_num_to_lookup.lower()}", f"article {article_num_to_lookup.upper()}", f"Article {article_num_to_lookup}"]
         if article_num_to_lookup.isdigit(): # Add numeric-only lookup if applicable
              article_key_variants.append(f"article {article_num_to_lookup}")

         for key in article_key_variants:
             if key in article_map:
                 print(f"Found direct article match in map: {key}. Retrieving specific text.")
                 # Return ONLY the direct match for verbatim requests or specific explanations
                 # For other types of queries, we might want to include semantic context too.
                 # Let's add the direct match to context_chunks but *also* proceed to semantic search
                 # unless it's clearly a verbatim text request.
                 # This is a design choice: pure direct lookup vs. hybrid.
                 # Let's stick to the previous logic: if direct article found, just return that unless it's empty.
                 if article_map[key].strip(): # Check if the retrieved text is not empty
                      print("Returning direct article text.")
                      return [article_map[key]]
                 else:
                      print(f"Warning: Direct article match found for {key} but text is empty. Proceeding to semantic search.")
                      # Don't add empty text to context_chunks, proceed to semantic search.
                      break # Exit loop, move to semantic search

         print(f"Direct lookup for Article {article_num_to_lookup} failed or returned empty text. Proceeding to semantic search.")


    # Check if the *current* question or the *follow-up reference* specifies an act section
    is_legal_act_query_curr = is_legal_act_query(question, act_names) # Checks for section number implicitly
    is_legal_act_query_followup = False
    section_num_followup = None
    act_name_followup = None

    if reference_text:
         # Check if the reference looks like a section number
         section_match_ref = re.search(r'\bsection\s+(\d+)([a-z]?)', reference_text.lower())
         if section_match_ref:
              is_legal_act_query_followup = True
              section_num_followup = section_match_ref.group(1) + (section_match_ref.group(2) if section_match_ref.group(2) else "")
              # Try to extract Act name from the reference_text if present
              for act_name_check in act_names:
                   cleaned_act_name_check = re.sub(r',\s*\d{4}$', '', act_name_check).strip().lower()
                   if re.search(r'\b' + re.escape(cleaned_act_name_check) + r'\b', reference_text.lower()):
                        act_name_followup = act_name_check
                        break


    # If a section is found either in the current query or the follow-up reference
    if is_legal_act_query_curr or is_legal_act_query_followup:
        # Get the section number and potentially act name from the current query first
        section_num_curr = None
        act_name_curr = None
        section_match_curr = re.search(r'\bsection\s+(\d+)([a-z]?)', lower_q)
        if section_match_curr:
             section_num_curr = section_match_curr.group(1) + (section_match_curr.group(2) if section_match_curr.group(2) else "")
             # Try to find an Act name in the *current* query
             for act_name_check in act_names:
                 cleaned_act_name_check = re.sub(r',\s*\d{4}$', '', act_name_check).strip().lower()
                 if re.search(r'\b' + re.escape(cleaned_act_name_check) + r'\b', lower_q):
                      act_name_curr = act_name_check
                      break


        section_num_to_lookup = section_num_curr if section_num_curr else section_num_followup
        act_name_to_lookup = act_name_curr if act_name_curr else act_name_followup

        if section_num_to_lookup:
             print(f"Attempting direct lookup for Section: {section_num_to_lookup} (Act: {act_name_to_lookup if act_name_to_lookup else 'any/from history'})")

             found_direct_section_match = False
             # Prioritize looking up with the specific act name if available
             if act_name_to_lookup:
                 print(f"Trying lookup with Act: {act_name_to_lookup}")
                 cleaned_act_name = re.sub(r',\s*\d{4}$', '', act_name_to_lookup).strip()
                 section_key_variants_with_act = [
                     f"{act_name_to_lookup.lower()} section {section_num_to_lookup.lower()}",
                     f"{cleaned_act_name.lower()} section {section_num_to_lookup.lower()}",
                     f"section {section_num_to_lookup.lower()} {act_name_to_lookup.lower()}",
                     f"section {section_num_to_lookup.lower()} {cleaned_act_name.lower()}"
                 ]
                 for key in section_key_variants_with_act:
                      if key in act_sections_map:
                          if act_sections_map[key].strip():
                               print(f"Found direct Act Section match: {key}. Retrieving specific text.")
                               return [act_sections_map[key]] # Direct match found, return only this
                          else:
                               print(f"Warning: Direct Act Section match found for {key} but text is empty.")
                          found_direct_section_match = True # Indicate we tried/found a match, even if empty
                          break # Stop looking with act name if a match was attempted

             # If no specific act name was provided or found, or lookup with act name failed, try looking up by section number alone
             # (This might return a section from a less relevant act if numbers overlap)
             if not found_direct_section_match:
                 print(f"Trying lookup by Section number alone: {section_num_to_lookup}")
                 # Loop through all section keys to find a match based on number alone
                 # Iterate over map items to get both key and value
                 for key, value in act_sections_map.items():
                     # Extract section number from map key
                     section_match_in_key = re.search(r'\bsection\s+(\d+)([a-z]?)', key.lower())
                     if section_match_in_key:
                         key_section_num = section_match_in_key.group(1) + (section_match_in_key.group(2) if section_match_in_key.group(2) else "")
                         if key_section_num.lower() == section_num_to_lookup.lower():
                              if value.strip():
                                   print(f"Found Section match (by number alone): {key}. Retrieving specific text.")
                                   # Returning the first match found by number alone
                                   return [value]
                              else:
                                   print(f"Warning: Section match by number alone found for {key} but text is empty.")
                              found_direct_section_match = True # Indicate we found a match by number

                 # If a section number was in the query/reference but *no* section context was found by direct lookup
                 if section_num_curr or section_num_followup: # Only print this if a section number was actually looked for
                      print(f"Direct lookup for Section {section_num_to_lookup} failed or returned empty text. Proceeding to semantic search.")
        else: # is_legal_act_query was true, but no section number found (only Act name mentioned)
             print(f"Legal act name detected ('{act_name_to_lookup}' from query/history), but no specific section number found. Proceeding to semantic search.")


    # If no direct Article or Section lookup returned context, perform semantic search
    if not context_chunks: # This condition is now only true if direct lookup failed or wasn't attempted
        print("Using semantic search...") # Console print
        try:
            # Ensure embedder is loaded (it's loaded in load_rag_data and passed here)
            if embedder is None:
                 print("❌ Embedder model is not loaded. Cannot perform semantic search.")
                 st.error("Embedding model is not loaded. Cannot perform semantic search.")
                 return [] # Return empty context if model is missing

            query_embedding = embedder.encode([question], convert_to_tensor=False).astype('float32') # Ensure numpy float32
            faiss.normalize_L2(query_embedding) # Normalize for Inner Product search
            k_semantic = 7 # Number of semantic chunks to retrieve
            # Ensure faiss_index is loaded and not None
            if faiss_index is None:
                 print("❌ FAISS index is not loaded. Cannot perform semantic search.")
                 st.error("FAISS index is not loaded. Cannot perform semantic search.")
                 return [] # Return empty context

            if faiss_index.ntotal == 0:
                 print("⚠️ FAISS index is empty. Cannot perform semantic search.")
                 st.warning("Search index is empty. Cannot perform semantic search.")
                 return [] # Return empty context

            # Clamp k_semantic to the total number of vectors in the index if needed
            search_k = min(k_semantic, faiss_index.ntotal)
            if search_k == 0: return [] # No vectors to search

            _, indices = faiss_index.search(np.array(query_embedding), search_k)

            # Retrieve unique chunks based on indices
            unique_chunks = set()
            for i in indices[0]:
                # Add bounds check and check for valid index (-1 can be returned sometimes)
                if 0 <= i < len(semantic_chunks):
                     semantic_chunk = semantic_chunks[i]
                     if semantic_chunk.strip(): # Only add non-empty chunks
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
        # Let's take the last 4 turns (question + answer pairs)
        recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
        history_entries = []
        for entry in recent_history:
            hist_q = entry.get('question', 'N/A')
            # Clean up potential Streamlit markdown from previous answers
            answer_text = str(entry.get('answer', ''))
            # Using re.sub with a non-greedy match and specifying count=1 for efficiency
            cleaned_answer_text = re.sub(r'🔍 \*\*Answer:\*\*(\n\n)?', '', answer_text, count=1)
            cleaned_answer_text = re.sub(r'❌ \*\*Error:\*\*(\n\n)?', '', cleaned_answer_text, count=1) # Also clean up error indicators
            cleaned_answer_text = cleaned_answer_text.strip()

            # Limit history length to avoid exceeding context window (rough estimate)
            max_hist_len_per_turn = 400 # characters per turn in history
            hist_entry_text = f"Previous Question: {hist_q}\nPrevious Answer: {cleaned_answer_text}"

            if len(hist_entry_text) > max_hist_len_per_turn:
                 # Simple truncation
                 hist_entry_text = hist_entry_text[:max_hist_len_per_turn - 10] + "..." # -10 for "...\n"

            history_entries.append(hist_entry_text)

        # Join history entries with a clear separator
        history_context = "\n\n--- Previous Conversation Context ---\n\n" + "\n\n---\n\n".join(history_entries) + "\n\n"

    context = "\n\n--- Retrieved Document Chunk ---\n\n".join(context_list)

    # --- Prompting Logic ---
    # Added more specific checks to prioritize direct text/explanation requests
    # based on the *current* question, even if the context isn't a perfect match,
    # as the user intent is clear.

    # Check if the user is asking for the exact text or title of a specific article/section
    is_direct_request = is_direct_text_request(question)
    is_explanation_request_flag = is_explanation_request(question)

    # Add instructions for the model about its persona and source
    persona_instruction = "You are an AI assistant specialized in Indian law, answering based *only* on the provided legal texts (Indian Constitution articles and Legal Act sections)."

    if is_direct_request:
        print("Direct text request detected. Using verbatim response prompt.")
        # Refine the direct text prompt
        return f"""{persona_instruction} Your primary task is to provide the EXACT text or title as it appears in the provided context when asked for specific articles or sections.

{history_context}
Carefully read the retrieved text below. This text may or may not contain the exact article or section you were asked about:
---------------------
{context}
---------------------

Instructions:
1. Identify if the user is asking for the *exact text* or the *title* of a specific Article or Section (e.g., "what does Article 21 state", "text of IPC Section 302", "title of Article 14").
2. Scan the provided context *only* for the requested Article/Section number and its content/title.
3. If the EXACT text or title for the requested Article or Section is present in the context and matches the user's request type (text vs. title), provide ONLY that exact text or title. Include the official article/section number and title exactly as shown in the text.
4. DO NOT paraphrase, explain, or interpret the content if a direct text or title request is made and the corresponding text/title is found in the context.
5. If the requested Article/Section text or title is NOT present in the provided context, politely mention that you don't have the specific text/title *in the context you received*. Then, share any information you *do* have from the context that might be related to the user's query or the mentioned article/section number, but be clear it's not the verbatim text/title requested.
6. If the context is empty or completely irrelevant, state that you cannot find relevant information in the provided text.

Question: {question}"""

    elif is_explanation_request_flag:
        print("Explanation request detected. Using detailed explanation prompt.")
        # Refine the explanation prompt
        return f"""{persona_instruction} Your task is to provide a DETAILED AND COMPREHENSIVE EXPLANATION of the relevant part of the law based *only* on the provided context.

{history_context}
Carefully read the retrieved text below. This text may or may not contain the exact article or section you were asked about, but should be related:
---------------------
{context}
---------------------

Instructions:
1. Read the provided context carefully to understand the legal provisions it contains, especially if specific articles or sections are mentioned.
2. Based *only* on the information in the context, explain the relevant legal concepts, clauses, terms, and implications in simple, accessible language.
3. If specific articles or sections are mentioned in the context and are relevant to the user's request (e.g., explaining Article 21), explain *what the context says* about them. Cite them by number (e.g., Article 21, IPC Section 302) if they are present in the context.
4. Use formatting (like bullet points, headings, bold text) to make your explanation clear and structured.
5. Provide context and practical significance *based on the provided text*.
6. Aim for a comprehensive explanation, drawing all relevant points *from the context*.
7. If the requested explanation (e.g., of a specific article) is not possible based *only* on the provided context, politely state that and share what information you *can* explain from the text you received that might be related.
8. If the context is empty or completely irrelevant, state that you cannot find relevant information to provide an explanation based on the provided text.

Question: {question}"""

    # Default RAG prompt for general questions or if direct text/explanation wasn't requested
    print("Using Standard RAG prompt for general query.")
    # Refine the standard RAG prompt
    return f"""{persona_instruction} Answer the user's question using *only* the information found in the provided context.

{history_context}
Carefully read the retrieved text below:
---------------------
{context}
---------------------

Based on the text provided, answer the following question: {question}

Instructions:
1. Answer the question clearly and concisely, using information found *only* in the context.
2. Use natural, conversational language unless quoting directly from the text.
3. Use phrases like "According to the text provided...", "Based on Article X...", or "Section Y states..." to refer to the source material in the context. Cite relevant Articles or Sections if mentioned in the context and relevant to the answer.
4. If you cannot find a complete answer *within the provided text*, state this politely. Share whatever relevant partial information *is* available in the context (e.g., "The provided text mentions Article 21 but does not directly answer your question about X"). DO NOT make up information or use outside knowledge.
5. If the context is empty or completely irrelevant, state that you cannot find relevant information to answer the question.
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
    if model is None:
         return "Error: Language model is not loaded."

    try:
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_output_tokens,
            temperature=0.2, # Slightly lower temperature for more focused legal answers
            top_p=0.85,
            top_k=40 # Slightly higher top_k
        )

        # Safety settings - typically more relaxed for RAG on legal text
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

st.set_page_config(page_title="Indian Law RAG Assistant", layout="wide") # Changed title slightly

st.title("⚖️ Indian Law RAG Assistant")
st.info("Ask questions about the Indian Constitution and legal acts based on the provided data.")

# Load RAG data and Language Model using cached functions
# These will run only once per Streamlit session
rag_system_data = load_rag_data() # This now loads data AND the embedding model
gemini_model = load_gemini_pro() # This reads from env var

# Check if setup was successful. If not, display errors and stop the app.
# Check both the RAG data (which now includes the embedder) and the Gemini model
if not rag_system_data or not gemini_model:
    st.error("🔴 Critical Error: Application setup failed. Please check the console output and the error messages above.")
    st.warning("Ensure all required data files are in the 'data' folder, the embedding model ('paraphrase-multilingual-mpnet-base-v2') can be downloaded, and that the 'GOOGLE_API_KEY' environment variable is correctly set.")
    st.stop() # Stop the Streamlit app execution

# Extract components for easier access (will always be available if st.stop() wasn't called)
# Ensure embedder is extracted correctly from the loaded data
faiss_index = rag_system_data['faiss_index']
embedder = rag_system_data['embedder'] # Extract the loaded embedder
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
         # Check if the answer is the placeholder before rendering
         if chat['answer'] == "Thinking...":
             st.info("Thinking...") # Use st.info for the spinner/thinking message
         else:
             st.markdown(chat['answer']) # Display the final answer with markdown


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
         # We don't need to store act_names per turn here, it's available globally
         'article_reference': None, # Placeholder for extracted reference
         'section_reference': None, # Placeholder for extracted reference
         'act_name_reference': None # Placeholder for extracted reference
     })
     # Note: The act_names list is needed by is_legal_act_query inside get_context.
     # It's loaded into the global `rag_system_data` dictionary, so it's accessible.
     st.rerun() # Rerun to display the user message and spinner state

# Logic to process the question if the last entry is the "Thinking..." placeholder
# This runs *after* the rerun triggered by submitting the question
# It ensures processing only happens once per user question.
if st.session_state.chat_history and st.session_state.chat_history[-1]['answer'] == "Thinking...":
    question_to_process = st.session_state.chat_history[-1]['question']

    # The spinner is now triggered by the "Thinking..." message display above.
    # We just need the processing logic here.

    try:
        # --- Core Logic: Retrieve Context, Create Prompt, Generate Answer ---

        print(f"\nUser Query: {question_to_process}") # Console print
        print("🔎 Retrieving relevant context...") # Console print

        # Pass history to get_context (exclude the current "Thinking..." turn)
        history_for_context = st.session_state.chat_history[:-1]

        # Check if embedder or faiss_index are None before calling get_context
        if embedder is None or faiss_index is None:
            answer = "❌ **Error:** Embedding model or search index is not loaded. Cannot process your query."
            print("❌ Cannot call get_context: Embedder or FAISS index is None.")
            st.session_state.chat_history[-1]['answer'] = answer
            st.rerun() # Update UI with error
            st.stop() # Stop further execution for this turn

        context = get_context(
            question_to_process,
            faiss_index,
            embedder,
            semantic_chunks,
            article_map,
            act_sections_map,
            act_names, # Pass act_names
            history_for_context # Pass previous turns
        )

        # --- Update history with extracted references *before* generating answer ---
        # Extract referenced article or section from the *original* question
        is_article, article_num = is_article_specific_query(question_to_process)
        if is_article:
            st.session_state.chat_history[-1]['article_reference'] = article_num
            print(f"Storing article reference for history: {article_num}")

        section_match = re.search(r'\bsection\s+(\d+)([a-z]?)', question_to_process.lower())
        if section_match:
            section_num = section_match.group(1) + (section_match.group(2) if section_match.group(2) else "")
            st.session_state.chat_history[-1]['section_reference'] = section_num
            print(f"Storing section reference for history: {section_num}")
            # Try to find act name in the original question too for history
            act_name_in_q = None
            for act_name_check in act_names:
                 cleaned_act_name_check = re.sub(r',\s*\d{4}$', '', act_name_check).strip().lower()
                 if re.search(r'\b' + re.escape(cleaned_act_name_check) + r'\b', question_to_process.lower()):
                      act_name_in_q = act_name_check
                      break
            if act_name_in_q:
                 st.session_state.chat_history[-1]['act_name_reference'] = act_name_in_q
                 print(f"Storing act name reference for history: {act_name_in_q}")


        # --- Generate Answer ---
        if not context:
            print("⚠️ No relevant context found for this query.") # Console print
            # Provide helpful suggestions based on available data
            available_articles = []
            # Collect article numbers from map keys
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

            articles_sample = ", ".join(available_articles[:20]) + ("..." if len(available_articles) > 20 else "") # Display sample articles

            # Collect Act names from the loaded list
            act_names_list = list(act_names)
            act_names_sample = ", ".join(act_names_list) + ("..." if len(act_names_list) > 5 else "") # Display sample act names


            answer = (
                f"🔍 **Answer:**\n\nI apologize, but I couldn't find enough relevant information in my current knowledge base to answer your question about '{question_to_process.strip()}'. "
                "My data primarily covers articles from the Indian Constitution and relevant legal acts.\n\n"
                "Could you try asking about a specific article number (e.g., Article 21), a legal act by name (e.g., Indian Penal Code), or a specific section within an act (e.g., IPC Section 302)?\n\n"
                f"Articles covered include: {articles_sample}\n\n"
                f"Legal Acts covered include: {act_names_sample}"
            )
            print("Generated 'No context found' answer.")


            # Update the last history entry with the final answer and no context
            st.session_state.chat_history[-1]['answer'] = answer
            st.session_state.chat_history[-1]['context'] = [] # No context for this case

        else: # Context was found
            print(f"✅ Retrieved {len(context)} context chunks for generation.") # Console print
            print("💭 Generating answer with Gemini...") # Console print

            # Pass history to create_prompt (exclude the current "Thinking..." turn)
            prompt = create_prompt(question_to_process, context, history_for_context) # Pass previous turns

            answer = generate_answer(gemini_model, prompt)
            formatted_answer = f"🔍 **Answer:**\n\n{answer}" # Add Markdown formatting
            print("Generated Gemini answer.")

            # Update the last history entry
            st.session_state.chat_history[-1]['answer'] = formatted_answer
            st.session_state.chat_history[-1]['context'] = context # Store retrieved context for potential inspection


        # Keep chat history to a reasonable size (e.g., last 10 interactions)
        # Pop from the beginning if history exceeds the limit
        while len(st.session_state.chat_history) > 10:
             st.session_state.chat_history.pop(0)
             print("Trimmed chat history.")

        # Rerun to update the display with the final answer and remove the spinner state
        st.rerun()

    except Exception as e:
        # Handle errors during retrieval or generation
        error_message = f"Sorry, an error occurred during processing: {str(e)}"
        st.error(error_message) # Display error in the UI
        st.exception(e) # Show traceback in the Streamlit UI (useful for debugging)

        # Update the last history entry with the error message
        st.session_state.chat_history[-1]['answer'] = f"❌ **Error:** {error_message}"
        st.session_state.chat_history[-1]['context'] = [] # Clear context on error
        st.session_state.chat_history[-1]['article_reference'] = None # Clear references on error
        st.session_state.chat_history[-1]['section_reference'] = None
        st.session_state.chat_history[-1]['act_name_reference'] = None
        print(f"Error processing query: {e}\n{traceback.format_exc()}")


        # Rerun to show the error message in the chat history
        st.rerun()


# Add a sidebar with instructions and examples
st.sidebar.title("About")
st.sidebar.info(
    "This AI-powered chatbot is built to assist users with fast, accurate legal information based on the Indian Constitution and important legal Acts such as the IPC, CrPC, Evidence Act, and more. Whether you're preparing for the AIBE (All India Bar Examination), studying law, or practicing as a legal professional, this tool is designed to simplify your legal research by delivering clear answers to your queries — one Article or one Act section at a time.\n\n"
    "The chatbot offers a user-friendly interface where you can ask questions like “What does Article 19 say?” or “Explain Section 300 of IPC.” It responds instantly using official legal texts, making it a valuable resource for revision, case preparation, and academic learning. Please note: to maintain precision and clarity, the chatbot supports only one Article or one Act-based question at a time."
)

st.sidebar.title("The CHAT-BOT has the access to data of all these:-")
st.sidebar.write("- Indian Constitution(article 1 - 395")
st.sidebar.write("- Indian Penal Code, 1860 (IPC)")
st.sidebar.write("- Contract Act, 1872")
st.sidebar.write("- Advocates Act, 1961")
st.sidebar.write("- Right to Information Act, 2005")
st.sidebar.write("- Consumer Protection Act, 2019")
st.sidebar.write("- Family Laws (Hindu Marriage Act, Hindu Succession Act, Muslim Personal Laws, etc.)")
st.sidebar.write("- Specific Relief Act, 1963")
st.sidebar.write("- Environment Protection Act, 1986") # Example follow-up
st.sidebar.write("- Intellectual Property Laws (Copyright, Trademark, Patent Acts)")
st.sidebar.write("- Bharatiya Nyaya Sanhita, 2023")
st.sidebar.write("- Bharatiya Nagarik Suraksha Sanhita, 2023")
st.sidebar.write("- Bharatiya Sakshya Adhiniyam, 2023") # Example section lookup without explicit act name (if section number is unique enough or in context)
# Example section lookup without explicit act name (if section number is unique enough or in context)
# Example section lookup without explicit act name (if section number is unique enough or in context)
