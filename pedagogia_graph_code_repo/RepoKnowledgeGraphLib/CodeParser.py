import logging
import os
from dotenv import load_dotenv
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from .utils.logger_utils import setup_logger
load_dotenv()


LOGGER_NAME = 'CODE_PARSER_LOGGER'
CODE_CHUNK_OVERLAP = int(os.getenv('CODE_CHUNK_OVERLAP', 0))
CODE_CHUNK_SIZE = int(os.getenv('CODE_CHUNK_SIZE', 2000))


class CodeParser:
    def __init__(self):
        setup_logger(LOGGER_NAME)
        self.logger = logging.getLogger(LOGGER_NAME)

        self.extension_mapping = {
            'c': Language.C,
            'h': Language.C,
            'cpp': Language.CPP,
            'cc': Language.CPP,
            'cxx': Language.CPP,
            'hpp': Language.CPP,
            'hh': Language.CPP,
            'hxx': Language.CPP,
            'go': Language.GO,
            'java': Language.JAVA,
            'py': Language.PYTHON,
            'pyw': Language.PYTHON,
            'js': Language.JS,
            'mjs': Language.JS,
            'cjs': Language.JS,
            'md': Language.MARKDOWN,
            'markdown': Language.MARKDOWN,
            'html': Language.HTML,
        }

    def parse(self, file_name:str, file_content:str) -> list:
        file_extension = file_name.split('.')[-1]

        try:
            self.logger.debug(f'Parsing file: {file_name}')
            if file_extension not in self.extension_mapping:
                self.logger.debug(f'File extension not supported: {file_extension}')
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CODE_CHUNK_SIZE,
                    chunk_overlap=CODE_CHUNK_OVERLAP,
                    length_function=len,
                    is_separator_regex=False,
                )
                docs = text_splitter.create_documents([file_content])

            else:
                self.logger.debug(f'File extension supported: {file_extension}')
                code_splitter = RecursiveCharacterTextSplitter.from_language(language=self.extension_mapping[file_extension], chunk_size=CODE_CHUNK_SIZE, chunk_overlap=CODE_CHUNK_OVERLAP)
                docs = code_splitter.create_documents([file_content])
        except Exception as e:
            self.logger.error(f'Error when parsing code: {e}')
        return [doc.page_content for doc in docs]





