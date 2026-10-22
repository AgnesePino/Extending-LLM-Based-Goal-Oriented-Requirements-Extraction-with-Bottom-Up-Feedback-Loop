# function to retreave the readme and the swagger APIs
import re
import requests
import json
from html.parser import HTMLParser
from html import unescape

from src.data_model import API

def get_markdown(link: str):
    res = requests.get(url=link)
    return res.text


class _HTMLTextExtractor(HTMLParser):
    """Extract only the text of an HTML fragment, ignoring script/style.

    Used to convert the HTML snapshot of a GitHub page into plain text, without
    adding an external dependency (e.g. BeautifulSoup) for such a narrow task.
    """

    _BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "pre"}

    def __init__(self):
        super().__init__()
        self._chunks = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        text = unescape("".join(self._chunks))
        # collapse the runs of blank lines left by block tags
        return re.sub(r"\n{3,}", "\n\n", text).strip()


_GITHUB_README_ARTICLE_RE = re.compile(
    r'<article class="markdown-body entry-content container-lg" itemprop="text">(.*?)</article>',
    re.S,
)


def get_local_markdown(path: str) -> str:
    """Read the README from a local snapshot instead of downloading it.

    The snapshot is the rendered GitHub HTML page, not the raw markdown. To keep the
    rest of the page (nav, sidebar, badges) out of the prompt, the `<article>` block
    that GitHub uses for the README content is isolated and only its text is
    extracted. The snapshot file itself is never modified.

    Args:
        path: path of the snapshot file.
    """
    with open(path, "r", encoding="utf-8") as fh:
        html = fh.read()

    match = _GITHUB_README_ARTICLE_RE.search(html)
    fragment = match.group(1) if match else html

    extractor = _HTMLTextExtractor()
    extractor.feed(fragment)
    return extractor.get_text()


def get_api_list_from_swagger(link):
    api_list = get_markdown(link)

    json_api_list = json.loads(api_list)["paths"]
    api_paths = json_api_list.keys()

    preprocessed_api_list = []

    for api in api_paths:
        path = json_api_list[api]
        for method in path.keys():
            preprocessed_api_list.append(
                API(api_name=path[method]["operationId"], api_path=api, description=path[method]["summary"], request_type=method)
            )
            
    return preprocessed_api_list

def api_list_to_string(api_list):
    apis = ""
    for api in api_list:
        apis += api.api_name + ", "
    # Remove the trailing comma and add a newline
    apis = apis.rstrip(", ") + "\n"
    return apis
