from dash import Dash, html, dcc 
import dash_bootstrap_components as dbc  
from dash.dependencies import Input, Output, State
import json
import numpy as np
import os
import string
from collections import defaultdict, Counter
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import math


def merged_inverted_indexes(file_paths):
    merged_index = defaultdict(dict)
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            inverted_index = json.load(file)
            for word, postings in inverted_index.items():
                for doc_id, positions in postings.items():
                    if doc_id not in merged_index[word]:
                        merged_index[word][doc_id] = []
                    merged_index[word][doc_id].extend(positions)
    
    return merged_index

file_paths = ['ihthizam_inverted_index.json', 'nuhan_inverted_index.json', 'abdur_inverted_index.json']
inverted_index = merged_inverted_indexes(file_paths)

num_documents = len({doc for docs in inverted_index.values() for doc in docs})
# Function to create the log term frequency for term in each document
def doc_log_tf(inverted_index, num_documents):
    tf_doc = defaultdict(dict)
    idf = {}

    for term, postings in inverted_index.items():
        doc_frequency = len(postings)
        idf[term] = np.log(num_documents / (doc_frequency + 1))

    for term, postings in inverted_index.items():
        for doc_id in postings:
            tf = len(postings[doc_id])
            log_tf = 1 + np.log(tf) if tf > 0 else 0
            tf_doc[doc_id][term] = log_tf
    
    return tf_doc, idf

# Create document vectors and store the first line
def create_document_vectors(tf_doc, all_terms):
    document_vectors = {}
    term_list = list(all_terms)
    document_lines = {}
    document_urls = {}

    for doc_id, terms in tf_doc.items():
        vector = np.zeros(len(term_list))
        for i, term in enumerate(term_list):
            if term in terms:
                vector[i] = terms[term]
        document_vectors[doc_id] = vector

        file_path = os.path.join('Documents', doc_id)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                first_line = file.readline().strip()
                document_lines[doc_id] = first_line

                if first_line.startswith("URL:"):
                    url = first_line.split("URL:", 1)[1].strip()
                    document_urls[doc_id] = url
                else:
                    document_urls[doc_id] = "No URL found"
        else:
            document_lines[doc_id] = "File not found"
            document_urls[doc_id] = "No URL found"

    return document_vectors, term_list, document_lines, document_urls

# QLM Language model for all documents 
def document_language_model(inverted_index, total_terms_collection):
    language_models = {}
    doc_lengths = defaultdict(int)

    for term, postings in inverted_index.items():
        for doc_id, posting in postings.items():
            tf = len(posting)
            doc_lengths[doc_id] += tf

    for term, postings in inverted_index.items():
        for doc_id, posting in postings.items():
            tf = len(posting)
            if doc_id not in language_models:
                language_models[doc_id] = {}
            language_models[doc_id][term] = tf / doc_lengths[doc_id]

    return language_models, doc_lengths

# QLM linear interpolation
def linear_interpolation_JM_smoothing(query_terms, language_models, doc_lengths, total_terms_collection, inverted_index, lambda_param=0.7):
    query_likelihoods = {}
    epsilon = 1e-10
    query_term_freq = Counter(query_terms)

    for doc_id, model in language_models.items():
        score = 0.0 
        doc_length = doc_lengths[doc_id]

        for term in query_terms:
            query_term_count = query_term_freq[term]
            doc_term_freq = model.get(term, 0)
            collection_term_freq = sum(len(postings) for postings in inverted_index.get(term, {}).values())

            p_w_given_C = collection_term_freq / total_terms_collection if total_terms_collection > 0 else 0
            
            if doc_length > 0 and p_w_given_C > 0:
                term_prob = (1 + (1 - lambda_param) / lambda_param * (doc_term_freq / (doc_length * p_w_given_C))) + epsilon
                score += query_term_count * math.log(term_prob)

        query_likelihoods[doc_id] = score

    return query_likelihoods

# Rank documents for VSM and QLM
def cosine_similarity_ranking(query_vector, document_vectors):
    scores = {}

    for doc_id, doc_vector in document_vectors.items():
        dot_product = np.dot(query_vector, doc_vector)
        norm_q = np.linalg.norm(query_vector)
        norm_d = np.linalg.norm(doc_vector)

        scores[doc_id] = dot_product / (norm_q * norm_d) if norm_q and norm_d else 0.0

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:50]

# Ranking for QLM model
def QLM_ranking(query_likelihoods):
    return sorted(query_likelihoods.items(), key=lambda x: x[1], reverse=True)[:50]

# Preprocessing and Query vector for VSM
def preprocess_query(query):
    query = query.lower().translate(str.maketrans('', '', string.punctuation))
    return query.split()

def compute_query_vector(query, idf, all_terms):
    preprocessed_query_terms = preprocess_query(query)
    query_vector = np.zeros(len(all_terms))
    term_list = list(all_terms)
    
    for term in preprocessed_query_terms:
        if term in idf:
            tf = preprocessed_query_terms.count(term)
            log_tf = 1 + np.log(tf) if tf > 0 else 0
            index = term_list.index(term)
            query_vector[index] = log_tf * idf[term]

    return query_vector

# Calling the functions
inverted_index = merged_inverted_indexes(file_paths)
num_documents = len({doc for docs in inverted_index.values() for doc in docs})
tf_doc, all_terms = doc_log_tf(inverted_index, num_documents)
document_vectors, term_list, document_lines, document_urls = create_document_vectors(tf_doc, all_terms)

total_terms_collection = sum(len(posting) for postings in inverted_index.values() for posting in postings.values())
language_models, doc_lengths = document_language_model(inverted_index, total_terms_collection)


# APP
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Discover the Innovation", id = 'title'))
        ]),

    dbc.Row([
        dbc.Col(html.P("Information Retrieval System", id = 'subtitle'))
        ]),

    dbc.Row([
        dbc.Col(html.P("Developed by: Abdur - 010 | Ihthizam - 012 | Nuhan - 013", id = 'author'))
        ]),

    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dcc.Input(id='query-input', type='text', placeholder='Search...', className="form-control"),
            dbc.Button([html.I(className="fa fa-search")], id='search-button', color='success')])], width=6)], justify='center'),

    dbc.Tabs([
        dbc.Tab(label="Vector Space Model", tab_id="vsm"),
        dbc.Tab(label="Query Likelihood Model", tab_id="qlm")], id="tabs", active_tab="vsm"),

    dbc.Row([
        dbc.Col(html.Div(id='search-results', className='mt-5',))], justify='center')
        ],fluid=True)


# Callback to handle search and display results for both VSM and QLM
@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks'), Input('tabs', 'active_tab')],
    [State('query-input', 'value')]
)
def update_output(n_clicks, active_tab, query):
    if n_clicks is None or not query:
        return html.P("Search anything to get results.", style={'color': '#6c757d', 'font-style': 'italic'})

    if active_tab == "vsm":
        tf_doc, idf = doc_log_tf(inverted_index, num_documents)
        query_vector = compute_query_vector(query, idf, term_list)
        ranked_documents = cosine_similarity_ranking(query_vector, document_vectors)
        title = "Vector Space Model - Top Results:"
    else:
        query_terms = preprocess_query(query)
        query_likelihoods = linear_interpolation_JM_smoothing(query_terms, language_models, doc_lengths, total_terms_collection, inverted_index, lambda_param=0.7)
        ranked_documents = QLM_ranking(query_likelihoods)
        title = "Query Likelihood Model - Top Results:"

    result_output = [html.H4(title, style={'color': '#2c3e50', 'fontWeight': 'bold'})]
    for doc_id, score in ranked_documents[:50]:  # Display top 50 results
        first_line = document_lines.get(doc_id, "First line not available")
        doc_url = document_urls.get(doc_id, "#")
        result_output.append(html.P([
            html.Span(f"Document: {doc_id} | Score: {score:.5f} ", style={'color': '#2c3e50', 'fontWeight': 'bold'}),
            html.Br(),
            html.Span(f"{first_line}", style={'color': '#6c757d'}),
            html.Br(),
            html.A("Read More...", href=doc_url, target="_blank", style={'color': '#007bff'})
        ], className='my-3'))

    return result_output

if __name__ == "__main__":
    app.run_server(debug=True)
