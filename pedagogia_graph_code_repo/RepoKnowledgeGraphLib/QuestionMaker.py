import logging 
import asyncio
from tqdm import tqdm

from .RepoKnowledgeGraph import RepoKnowledgeGraph
from .ModelService import ModelService
from .utils.logger_utils import setup_logger
from .utils.chunk_utils import organize_chunks_by_file_name, join_organized_chunks, extract_filename_from_chunk
from .Node import ChunkNode

LOGGER_NAME = "QUESTION_MAKER_LOGGER"

class QuestionMaker: 
    """
    The QuestionMaker class is responsible for generating code comprehension questions and answers
    based on code chunks and knowledge graphs. It leverages a language model service to formulate
    questions and answers that test deep understanding of code, focusing on mechanisms, design decisions,
    and subtle behaviors. It supports generating questions for neighboring code chunks as well as for
    specific entities (e.g., functions, classes) that are both declared and called in the codebase.
    """
    def __init__(self): 
        """
        Initializes the QuestionMaker, sets up logging, and instantiates the model service.
        """
        setup_logger(LOGGER_NAME)
        self.logger = logging.getLogger(LOGGER_NAME)
        self.model_service = ModelService()
        

    def generate_questions_answers(self, candidate_chunks:dict) -> list:
        """
        Placeholder for generating questions and answers from candidate chunks.
        Args:
            candidate_chunks (dict): Dictionary mapping chunk groups to process.
        Returns:
            list: List of question-answer pairs.
        """
        pass 
    
    def test_chunk_sensibility(self, knowledge_graph: RepoKnowledgeGraph) -> list:
        """
        Placeholder for testing the sensibility of code chunks in the knowledge graph.
        Args:
            knowledge_graph (RepoKnowledgeGraph): The knowledge graph to test.
        Returns:
            list: List of results or metrics.
        """
        pass 
    
 
    async def make_n_neighbouring_chunk_questions_async(self, knowledge_graph: RepoKnowledgeGraph) -> list:
        """
        Generates questions and answers for all possible groups of n directly neighboring code chunks
        in each file of the knowledge graph. This helps assess understanding of code that spans multiple
        adjacent chunks, such as related functions or code blocks.

        Args:
            knowledge_graph (RepoKnowledgeGraph): The knowledge graph to generate questions from.
        Returns:
            list: A list of dictionaries, each containing a question, answer, the involved chunks, and category.
        """
        file_nodes = knowledge_graph.get_all_files()
        # create candidate chunks dictionary
        candidate_chunks = []
        for file_node in file_nodes:
            self.logger.info(f"Processing file node: {file_node}")
            chunks = knowledge_graph.get_chunks_of_file(file_node.id)
            num_chunks = len(chunks)
            # For each n, collect all n-sized tuples of directly neighbouring chunks
            for n in range(2, num_chunks + 1):
                for i in range(num_chunks - n + 1):
                    # Only directly neighbouring chunks
                    candidate_chunks.append(list(chunks[i:i+n]))
        # generate questions and answers from candidate chunks in parallel, in batches of 15
        
        async def process_chunk_group(chunks):
            """
            Helper coroutine to generate a question and answer for a specific group of neighboring chunks.
            Args:
                chunks (list): The list of code chunks to generate the question and answer from.
            Returns:
                dict: Contains question, answer, chunks, and category.
            """
            question = await self._generate_neighboring_question_from_chunks_async(chunks)
            answer = await self.answer_question_about_chunks_async(chunks, question)
            return {
                'question': question,
                'clean_question': question,
                'answer': answer,
                'chunks': [chunk.dict() for chunk in chunks], 
                'category': 'neighbouring_chunks'
            }
        # Batch processing in groups of 15 with tqdm
        batch_size = 15
        results = []
        total = len(candidate_chunks)
        for i in tqdm(range(0, total, batch_size), desc="Generating neighbouring chunk questions", unit="batch"):
            batch = candidate_chunks[i:i+batch_size]
            tasks = [process_chunk_group(chunks) for chunks in batch]
            batch_results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Questions in batch", leave=False):
                batch_results.append(await coro)
            results.extend(batch_results)
        return results

    async def make_entity_declaration_call_specific_questions_async(self, knowledge_graph: RepoKnowledgeGraph) -> list:
        """
        Generates questions and answers about specific entities (e.g., functions, classes) that have both
        a declaration and at least one call site in the knowledge graph. Focuses on cross-file references
        by default.

        Args:
            knowledge_graph (RepoKnowledgeGraph): The knowledge graph to generate questions from.
        Returns:
            list: A list of dictionaries, each containing a question, answer, entity, involved chunks, and category.
        """
        self.logger.info("Generating entity-specific questions.")
        candidate_pairs = self.get_entities_with_declaration_and_calling(knowledge_graph)
        
        async def process_entity_pair(pair):
            """
            Helper coroutine to generate a question and answer for a specific entity's declaration and call site.
            Args:
                pair (dict): Contains entity name, declaring_chunk_id, and calling_chunk_id.
            Returns:
                dict: Contains question, answer, entity, chunks, and category.
            """
            entity_name = pair['entity']
            chunks = [knowledge_graph[pair['declaring_chunk_id']], knowledge_graph[pair['calling_chunk_id']]]
            question = await self.make_entity_specific_question_async(chunks, entity_name)
            answer = await self.answer_question_about_chunks_async(chunks, question)
            return {
                'question': question,
                'clean_question': question,
                'answer': answer,
                'entity': entity_name,
                'chunks': [chunk.dict() for chunk in chunks], 
                'category': 'entity_declaration_call_specific'
            }
        
        # Batch processing with tqdm
        batch_size = 15
        results = []
        total = len(candidate_pairs)
        for i in tqdm(range(0, total, batch_size), desc="Generating entity-specific questions", unit="batch"):
            batch = candidate_pairs[i:i+batch_size]
            tasks = [process_entity_pair(pair) for pair in batch]
            batch_results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Questions in batch", leave=False):
                batch_results.append(await coro)
            results.extend(batch_results)
        return results
    
    async def make_interacting_entities_specific_questions_async(self, entity_A:str, entity_B:str, 
                                                            decl_chunk_A: ChunkNode, decl_chunk_B: ChunkNode, 
                                                            call_chunk: ChunkNode) -> str:
        """
        Generates a question and answer about two entities that interact in the same chunk.
        Each entity has a declaration and at least one call site, and the question focuses on their interaction.

        Args:
            entity_A (str): Name of the first entity.
            entity_B (str): Name of the second entity.
            decl_chunk_A (str): Chunk of the declaration of entity A.
            decl_chunk_B (str): Chunk of the declaration of entity B.
            call_chunk (str): Chunk ID where both entities interact.
        Returns:
            str: the generated question as plain text.
        """
        entity_A_definition_code = decl_chunk_A.content 
        entity_B_definition_code = decl_chunk_B.content
        entity_interaction_code = call_chunk.content
        
        prompt = f"""You are given two code entities, {entity_A} and {entity_B}, along with a snippet where they interact.  
        Your task is to write **one clear and concise question** about their relationship.  

        ### Input:
        * {entity_A} Definition Code:
        {entity_A_definition_code}

        * {entity_B} Definition Code:
        {entity_B_definition_code}

        * Interaction Code (where they interact):
        {entity_interaction_code}

        ### Guidelines:
        * Ask about design, abstraction, dependencies, or side effects.  
        * The question should highlight something a developer might consider when reviewing or improving the code.  
        * Keep the question short and direct so it can be answered briefly.  
        * Do not explain the code or provide answers.  

        ### Output:
        **Question**: <your question here>  
        """

        
        initial_question = await self.model_service.query_async(prompt=prompt)
        return await self.extract_question_from_generated_text_async(generated_text=initial_question)
    
    def get_all_candidate_pairs_and_triplets(self, knowledge_graph: RepoKnowledgeGraph) -> list:
        
        candidate_triplets = []
        candidate_pairs = []
        
        interacting_entity_triplets = self.get_interacting_entity_triplets(knowledge_graph)
        for triplet in interacting_entity_triplets:
            chunks = [
                    knowledge_graph[triplet['decl_chunk_A']],
                    knowledge_graph[triplet['decl_chunk_B']],
                    knowledge_graph[triplet['call_chunk']]
                ]
            candidate_triplets.append({
                'entities': (triplet['entity_A'], triplet['entity_B']),
                'chunks': [chunk.dict() for chunk in chunks],
                'category': 'interacting_entities'
            })
        
        declaration_calling_pairs = self.get_entities_with_declaration_and_calling(knowledge_graph)
        for pair in declaration_calling_pairs:
            chunks = [knowledge_graph[pair['declaring_chunk_id']], knowledge_graph[pair['calling_chunk_id']]]
            candidate_pairs.append({
                'entity': pair['entity'],
                'chunks': [chunk.dict() for chunk in chunks], 
                'category': 'entity_declaration_call_specific'
            })
        
        return candidate_pairs, candidate_triplets
        
        
    
    async def make_interacting_entity_questions_async(self, knowledge_graph: RepoKnowledgeGraph) -> list:
        """
        Generates questions and answers about pairs of entities that interact in the same chunk.
        Each entity has a declaration and at least one call site, and the question focuses on their interaction.

        Args:
            knowledge_graph (RepoKnowledgeGraph): The knowledge graph to generate questions from.
        Returns:
            list: A list of dictionaries, each containing a question, answer, entities, involved chunks, and category.
        """
        self.logger.info("Generating interacting entity questions.")
        triplets = self.get_interacting_entity_triplets(knowledge_graph)
        
        async def process_triplet(triplet):
            """
            Helper coroutine to generate a question and answer for a specific interacting entity triplet.
            Args:
                triplet (dict): Contains entity_A, entity_B, decl_chunk_A, decl_chunk_B, and call_chunk.
            Returns:
                dict: Contains question, answer, entities, chunks, and category.
            """
            chunks = [
                knowledge_graph[triplet['decl_chunk_A']],
                knowledge_graph[triplet['decl_chunk_B']],
                knowledge_graph[triplet['call_chunk']]
            ]
            question = await self.make_interacting_entities_specific_questions_async(entity_A=triplet['entity_A'],
                                                                                     entity_B=triplet['entity_B'],
                                                                                    decl_chunk_A=knowledge_graph[triplet['decl_chunk_A']], 
                                                                                    decl_chunk_B=knowledge_graph[triplet['decl_chunk_B']], 
                                                                                    call_chunk=knowledge_graph[triplet['call_chunk']])
            answer = await self.answer_question_about_chunks_async(chunks, question)
            return {
                'question': question,
                'clean_question': question, 
                'answer': answer,
                'entities': (triplet['entity_A'], triplet['entity_B']),
                'chunks': [chunk.dict() for chunk in chunks],
                'category': 'interacting_entities'
            }
        
        # Batch processing with tqdm
        batch_size = 15
        results = []
        total = len(triplets)
        for i in tqdm(range(0, total, batch_size), desc="Generating interacting entity questions", unit="batch"):
            batch = triplets[i:i+batch_size]
            tasks = [process_triplet(triplet) for triplet in batch]
            batch_results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Questions in batch", leave=False):
                batch_results.append(await coro)
            results.extend(batch_results)
        return results
    
    async def _generate_neighboring_question_from_chunks_async(self, chunks: list) -> str:
        """
        Generates a single code comprehension question for a group of code chunks using the model service.
        The question is designed to probe deep understanding of the code's mechanisms, design, or pitfalls.

        Args:
            chunks (list): The list of code chunks to generate the question from.
        Returns:
            str: The generated question as plain text.
        """
        organized_chunks = organize_chunks_by_file_name(chunks)
        joined_chunks = join_organized_chunks(organized_chunks)
        
        system_prompt = """You are an expert in evaluating code comprehension. The user will provide, in the next message, the content of a code submission (in any programming language). Your goal is to analyze this code, identify its critical, subtle, or obscure aspects, and generate **one relevant question in English** to ask someone in order to assess their understanding of the code.

        This question should focus on:

        * essential mechanisms of how the code works,
        * important design decisions,
        * potential pitfalls or unexpected behaviors,
        * or any aspect that requires deep comprehension.

        The goal is to test whether the person has **truly understood** the code—not just skimmed through it.

        Respond with **only one question**, in plain text. Do not include any explanation, comment, or wrapper (e.g., no dictionaries, no lists).
        """
        initial_question = await self.model_service.query_with_instructions_async(instructions=system_prompt, prompt=joined_chunks)
        return await self.extract_question_from_generated_text_async(generated_text=initial_question)
        
    async def answer_question_about_chunks_async(self, chunks: list, question: str) -> str:
        """
        Generates an answer to a code comprehension question about a group of code chunks using the model service.
        The answer should demonstrate deep understanding and cover mechanisms, design, and pitfalls.

        Args:
            chunks (list): The list of code chunks to answer the question about.
            question (str): The question to answer.
        Returns:
            str: The generated answer as plain text.
        """
        organized_chunks = organize_chunks_by_file_name(chunks)
        joined_chunks = join_organized_chunks(organized_chunks)
        
        system_prompt = """You are an expert in evaluating code comprehension. The user will provide, in the next message, the content of a code submission (in any programming language) and a question about it. Your goal is to analyze this code, identify its critical, subtle, or obscure aspects, and generate **one relevant answer in English** to the question.

        This answer should focus on:

        * essential mechanisms of how the code works,
        * important design decisions,
        * potential pitfalls or unexpected behaviors,
        * or any aspect that requires deep comprehension.
        The goal is to provide a clear and thorough answer that demonstrates a deep understanding of the code.
        """
        
        return await self.model_service.query_with_instructions_async(instructions=system_prompt, prompt=joined_chunks + "\n\n" + question)
          
    async def make_entity_specific_question_async(self, chunks: list, entity_name:str): 
        """
        Generates a question about a specific entity (e.g., function, class) in the context of the provided code chunks.
        The question is designed to probe understanding of the entity's purpose, behavior, and interactions.

        Args:
            chunks (list): The list of code chunks to generate the question from.
            entity_name (str): The name of the entity to focus on.
        Returns:
            str: The generated question as plain text.
        """
        organized_chunks = organize_chunks_by_file_name(chunks)
        joined_chunks = join_organized_chunks(organized_chunks)
        
        system_prompt = f"""You will be given one or more code snippets, possibly from multiple files.  

        A specific entity (such as a class, function, or variable) will be identified.  

        ---

        ## Entity of Focus: {entity_name}  

        ### Task:
        * Write **one clear and concise question** about this entity.  
        * The question should highlight something a developer might consider, such as its purpose, behavior, interactions, or potential improvements.  

        ### Guidelines:
        * Keep the question short and direct.  
        * Do not explain the code or give an answer.  

        ### Output:
        **Question**: <your question here>  
        """
        
        initial_question= await self.model_service.query_with_instructions_async(instructions=system_prompt, prompt=joined_chunks)
        return await self.extract_question_from_generated_text_async(generated_text=initial_question)
    
    def get_entities_with_declaration_and_calling(self, knowledge_graph: RepoKnowledgeGraph, cross_file_only: bool = True) -> list:
        """
        Finds all entities in the knowledge graph that have both a declaration and at least one call site.
        Optionally restricts to cases where the declaration and call are in different files (cross-file).

        Args:
            knowledge_graph (RepoKnowledgeGraph): The knowledge graph to search in.
            cross_file_only (bool): If True, only consider cross-file declaration/call pairs.
        Returns:
            list: List of dictionaries with 'entity', 'declaring_chunk_id', and 'calling_chunk_id'.
        """
        candidate_pairs = []
        entities = knowledge_graph.entities
        for entity_name in entities:
            entity = entities[entity_name]
            if len(entity['declaring_chunk_ids']) and len(entity['calling_chunk_ids']): 
                found = False
                for declaring_chunk_id in entity['declaring_chunk_ids']:
                    for calling_chunk_id in entity['calling_chunk_ids']:
                        if declaring_chunk_id != calling_chunk_id:
                            if cross_file_only and extract_filename_from_chunk(knowledge_graph[declaring_chunk_id]) == extract_filename_from_chunk(knowledge_graph[calling_chunk_id]):
                                continue
                            else: 
                                candidate_pairs.append({'entity': entity_name, 'declaring_chunk_id' : declaring_chunk_id, 'calling_chunk_id':  calling_chunk_id})
                                found = True
                                break
                    if found:
                        break
        return candidate_pairs

    def get_interacting_entity_triplets(self, knowledge_graph: RepoKnowledgeGraph) -> list:
        """
        Finds triplets of chunk ids such that:
        - Two entities (A, B) are interacting in the same chunk (call_chunk)
        - Each entity has a declaring chunk (decl_chunk_A, decl_chunk_B)
        - Both entities have non-empty declaring_chunk_ids and calling_chunk_ids

        Returns:
            list of dicts with keys:
                'entity_A', 'entity_B', 'decl_chunk_A', 'decl_chunk_B', 'call_chunk'
        """
        triplets = []
        seen_pairs = set()
        entities = knowledge_graph.entities
        for entity_A_name, entity_A in entities.items():
            if not entity_A['declaring_chunk_ids'] or not entity_A['calling_chunk_ids']:
                continue
            for entity_B_name, entity_B in entities.items():
                if entity_A_name == entity_B_name:
                    continue
                if not entity_B['declaring_chunk_ids'] or not entity_B['calling_chunk_ids']:
                    continue
                pair_key = (entity_A_name, entity_B_name)
                if pair_key in seen_pairs:
                    continue
                # Find intersection of calling_chunk_ids
                call_chunks = set(entity_A['calling_chunk_ids']) & set(entity_B['calling_chunk_ids'])
                found = False
                for call_chunk in call_chunks:
                    for decl_chunk_A in entity_A['declaring_chunk_ids']:
                        for decl_chunk_B in entity_B['declaring_chunk_ids']:
                            triplets.append({
                                'entity_A': entity_A_name,
                                'entity_B': entity_B_name,
                                'decl_chunk_A': decl_chunk_A,
                                'decl_chunk_B': decl_chunk_B,
                                'call_chunk': call_chunk
                            })
                            seen_pairs.add(pair_key)
                            found = True
                            break
                        if found:
                            break
                    if found:
                        break
        return triplets
    
    
    async def extract_question_from_generated_text_async(self, generated_text: str) -> str:
        """
        Extracts the question from the generated text. The question is expected to be the last line of the text.

        Args:
            generated_text (str): The text generated by the model.
        Returns:
            str: The extracted question.
        """
        
        prompt = f"Extract only the question from the following text. Return the question exactly, with no extra words or labels:\n\n{generated_text}\n\n"
        return await self.model_service.query_async(prompt=prompt)
    
    def select_diverse_candidates(self, candidate_pairs, candidate_triplets, max_pairs=20, max_triplets=20):
        """
        Selects a limited number of pairs and triplets with maximum diversity in entity representation.
        Args:
            candidate_pairs (list): List of candidate pairs (dicts with 'entity', ...).
            candidate_triplets (list): List of candidate triplets (dicts with 'entities', ...).
            max_pairs (int): Maximum number of pairs to select.
            max_triplets (int): Maximum number of triplets to select.
        Returns:
            (list, list): Selected pairs and triplets.
        """
        # Select pairs
        selected_pairs = []
        used_entities = set()
        for pair in candidate_pairs:
            entity = pair['entity']
            if entity not in used_entities:
                selected_pairs.append(pair)
                used_entities.add(entity)
            if len(selected_pairs) >= max_pairs:
                break
        # Select triplets
        selected_triplets = []
        used_entities_triplets = set()
        for triplet in candidate_triplets:
            entities = set(triplet['entities'])
            if not entities & used_entities_triplets:
                selected_triplets.append(triplet)
                used_entities_triplets.update(entities)
            if len(selected_triplets) >= max_triplets:
                break
        return selected_pairs, selected_triplets
    
    async def transform_answser_into_mcq_answer_async(self, question, answer, chunks):
        """
        Transforms the question and answer into a format suitable for MCQ generation.
        """
        code = join_organized_chunks(organize_chunks_by_file_name(chunks))  
        
        prompt = f"""
    You are an expert Python developer and technical writer. I will give you:

    1. A Python code snippet  
    2. A question about that code  
    3. A detailed answer to the question  

    Your task is to **sanitize** the answer. That means:

    - Strip away all fluff, filler, and redundant explanation  
    - Focus only on what directly answers the question  
    - Make it **short, clear, and direct**, as if it were a correct MCQ answer  
    - Prefer concise phrases or a single clear sentence over paragraph explanations  
    - Keep any necessary technical detail, but no more than needed  

    Do **not** repeat the question. Do **not** rephrase the code. Just give the concise, final answer.

    - **Input Code**:  
    {code}

    - **Question**:  
    {question}

    - **Original Answer**:  
    {answer}

    - **Sanitized Answer**:
    """
        return await self.model_service.query_async(prompt)

