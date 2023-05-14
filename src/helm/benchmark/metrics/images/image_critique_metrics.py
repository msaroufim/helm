from typing import Dict, List

import numpy as np

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.metrics.metric import Metric, MetricResult
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat, merge_stat
from helm.common.critique_request import CritiqueTaskTemplate, CritiqueQuestionTemplate, CritiqueRequest, QuestionType
from helm.common.images_utils import encode_base64, filter_blacked_out_images
from helm.common.hierarchical_logger import hlog
from helm.common.request import RequestResult
from .image_metrics_utils import gather_generated_image_locations


class ImageCritiqueMetric(Metric):
    """
    Critique evaluation for image generation. Possesses the ability to ask human
    annotators the following questions about the generated images:

    1. Image-text alignment
    2. If the subject of the image is clear (for aesthetics)
    3. How aesthetically pleasing the image is?
    4. How original the image is?
    5. If there are any possible copyright infringements (originality)?
    """

    ALIGNMENT_NAME: str = "image_text_alignment_human"
    ALIGNMENT_ANSWER_TO_SCORE: Dict[str, int] = {
        "Does not match at all": 1,
        "Has significant discrepancies": 2,
        "Has several minor discrepancies": 3,
        "Has a few minor discrepancies": 4,
        "Matches exactly": 5,
    }

    SUBJECT_NAME: str = "clear_subject_human"
    SUBJECT_ANSWER_TO_SCORE: Dict[str, int] = {
        "No, it's unclear.": 1,
        "I don't know. It's hard to tell.": 2,
        "Yes, it's clear.": 3,
    }

    AESTHETICS_NAME: str = "aesthetics_human"
    AESTHETICS_ANSWER_TO_SCORE: Dict[str, int] = {
        "I find the image ugly.": 1,
        "The image has a lot of flaws, but it's not completely unappealing.": 2,
        "I find the image neither ugly nor aesthetically pleasing.": 3,
        "The image is aesthetically pleasing and nice to look at it.": 4,
        "The image is aesthetically stunning. I can look at it all day.": 5,
    }

    ORIGINALITY_NAME: str = "originality_human"
    ORIGINALITY_ANSWER_TO_SCORE: Dict[str, int] = {
        "I’ve seen something like this before to the point it’s become tiresome.": 1,
        "The image is not really original, but it has some originality to it.": 2,
        "Neutral.": 3,
        "I find the image to be fresh and original.": 4,
        "I find the image to be extremely creative and out of this world.": 5,
    }

    COPYRIGHT_NAME: str = "copyright_human"
    NONE_ANSWER: str = "none"

    def __init__(
        self,
        study_title: str,
        include_alignment: bool,
        include_aesthetics: bool,
        include_originality: bool,
        include_copyright: bool,
        num_examples: int,
        num_respondents: int,
    ) -> None:
        self._study_title: str = study_title
        self._include_alignment: bool = include_alignment
        self._include_aesthetics: bool = include_aesthetics
        self._include_originality: bool = include_originality
        self._include_copyright: bool = include_copyright
        self._num_examples: int = num_examples
        self._num_respondents: int = num_respondents

    def __repr__(self) -> str:
        return "ImageCritiqueMetric()"

    def evaluate(
        self,
        scenario_state: ScenarioState,
        metric_service: MetricService,
        eval_cache_path: str,
        parallelism: int,
    ) -> MetricResult:
        request_states: List[RequestState] = scenario_state.request_states
        np.random.seed(0)
        if self._num_examples < len(request_states):
            request_states = list(
                np.random.choice(
                    request_states,  # type: ignore
                    self._num_examples,
                    replace=False,
                )
            )

        all_stats: Dict[MetricName, Stat] = {}
        for request_state in request_states:
            stats = self.evaluate_generation(
                scenario_state.adapter_spec,
                request_state,
                metric_service,
                eval_cache_path,
            )
            for stat in stats:
                merge_stat(all_stats, stat)

        return MetricResult(aggregated_stats=list(all_stats.values()), per_instance_stats=[])

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        request_result: RequestResult = request_state.result
        image_locations: List[str] = gather_generated_image_locations(request_result)
        image_locations = filter_blacked_out_images(image_locations)
        if len(image_locations) == 0:
            return []

        # Randomly select one of the generated images to critique
        image_path: str = np.random.choice(image_locations)
        prompt: str = request_state.request.prompt

        # Send the critique request
        template: CritiqueTaskTemplate = self._get_critique_template(adapter_spec.model)
        request = CritiqueRequest(template, fields={"prompt": prompt, "image": encode_base64(image_path)})
        result = metric_service.make_critique_request(request)
        if not result or not result.responses:
            return []

        if len(result.responses) == 0:
            # Skip computing metrics if there aren't any responses yet
            hlog("Waiting for responses to be collected.")
            return []

        stats: Dict[str, Stat] = {}
        for question in template.questions:
            stats[question.name] = Stat(MetricName(question.name))

        for response in result.responses:
            for answer_name, answer in response.answers.items():
                assert isinstance(answer, str)

                answer_value: float
                if answer_name == self.ALIGNMENT_NAME:
                    answer_value = self.ALIGNMENT_ANSWER_TO_SCORE[answer]
                elif answer_name == self.SUBJECT_NAME:
                    answer_value = self.SUBJECT_ANSWER_TO_SCORE[answer]
                elif answer_name == self.AESTHETICS_NAME:
                    answer_value = self.AESTHETICS_ANSWER_TO_SCORE[answer]
                elif answer_name == self.ORIGINALITY_NAME:
                    answer_value = self.ORIGINALITY_ANSWER_TO_SCORE[answer]
                elif answer_name == self.COPYRIGHT_NAME:
                    urls: List[str] = answer.split("\n")
                    has_copyright_infringement: bool = False
                    for url in urls:
                        url = url.strip()
                        if len(url) == 0:
                            continue

                        if url.lower() != self.NONE_ANSWER.lower():
                            has_copyright_infringement = True
                            hlog(f"Found possible infringement: {url}")
                    answer_value = 1 if has_copyright_infringement else 0
                else:
                    raise ValueError(f"Invalid answer type: {answer_name}")

                stats[answer_name].add(answer_value)
        return list(stats.values())

    def _get_critique_template(self, model_name: str) -> CritiqueTaskTemplate:
        hlog(f"Generating critique template for model: {model_name}")
        questions: List[CritiqueQuestionTemplate] = []

        if self._include_alignment:
            questions.append(
                CritiqueQuestionTemplate(
                    name=self.ALIGNMENT_NAME,
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    text="How well does the image match the description?",
                    options=list(self.ALIGNMENT_ANSWER_TO_SCORE.keys()),
                )
            )
        if self._include_originality:
            questions.append(
                CritiqueQuestionTemplate(
                    name=self.ORIGINALITY_NAME,
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    text="How <u>original</u> is the image, given it was created with the description?",
                    options=list(self.ORIGINALITY_ANSWER_TO_SCORE.keys()),
                )
            )
        if self._include_aesthetics:
            questions.extend(
                [
                    CritiqueQuestionTemplate(
                        name=self.SUBJECT_NAME,
                        question_type=QuestionType.MULTIPLE_CHOICE,
                        text="Is it clear who the subject(s) of the image is? The subject can be a living being "
                        "(e.g., a dog or a person) or an inanimate body or object (e.g., a mountain).",
                        options=list(self.SUBJECT_ANSWER_TO_SCORE.keys()),
                    ),
                    CritiqueQuestionTemplate(
                        name=self.AESTHETICS_NAME,
                        question_type=QuestionType.MULTIPLE_CHOICE,
                        text="How aesthetically pleasing is the image?",
                        options=list(self.AESTHETICS_ANSWER_TO_SCORE.keys()),
                    ),
                ]
            )
        if self._include_copyright:
            questions.append(
                CritiqueQuestionTemplate(
                    name=self.COPYRIGHT_NAME,
                    question_type=QuestionType.FREE_RESPONSE,
                    text="<p>Please follow the instructions carefully:</p>"
                    '1. Right click the image above and select "Search Image with Google”, which will open a '
                    "sidebar with Google Lens results.<br>"
                    "2. Adjust the bounding box to fit the entire image if necessary.<br>"
                    "3. Only for the first page of results, look for images that appear to be <b>almost identical</b> "
                    "to the image above to identify <b>potential copyright infringements</b>. For those images, "
                    "click on the image, which will open a new tab, and copy the URL for that tab.<br>"
                    "4. List the URLs from step 3 below. <b>If there are multiple URLs, list each on a new line.</b> "
                    f"If there are no URLs, answer <b>{self.NONE_ANSWER}</b><br>",
                    options=[],
                )
            )

        return CritiqueTaskTemplate(
            # name=f"{self._study_title},{model_name}",
            name="vhelm_image_critique",
            instructions="<p>Please answer the questions below about the following image and description.</p>"
            '<br><img src="data:image;base64,{{image}}"><br><p>Description: <b>{{prompt}}</b></p><br>',
            num_respondents=self._num_respondents,
            questions=questions,
        )
