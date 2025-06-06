# SPDX-License-Identifier: Apache-2.0

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Mapping, Optional
import time
import numpy as np

from vllm.beam_search import BeamSearchSequence, create_sort_beams_key_function
from vllm.config import DecodingConfig, ModelConfig, VllmConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.inputs.parse import is_explicit_encoder_decoder_prompt
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import Device, collect_from_async_generator, random_uuid
from vllm.sampling_params import RequestOutputKind

logger = init_logger(__name__)


class EngineClient(ABC):
    """Protocol class for Clients to Engine"""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_stopped(self) -> bool:
        ...

    @property
    @abstractmethod
    def errored(self) -> bool:
        ...

    @property
    @abstractmethod
    def dead_error(self) -> BaseException:
        ...

    @abstractmethod
    def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    @abstractmethod
    def generate_beam_search(
        self,
        prompt: List[PromptType],
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    @abstractmethod
    def generate_beam_search_post(
        self,
        prompt: List[PromptType],
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    async def beam_search_bak(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        preprocessor = await self.get_input_preprocessor()
        tokenizer_group = preprocessor.get_tokenizer_group()
        tokenizer = await tokenizer_group.get_lora_tokenizer_async()

        if is_explicit_encoder_decoder_prompt(prompt):
            raise NotImplementedError
        else:
            processed_inputs = preprocessor._prompt_to_llm_inputs(prompt)

        prompt_token_ids = processed_inputs["prompt_token_ids"]
        prompt_text = processed_inputs.get("prompt")
        multi_modal_data = processed_inputs.get("multi_modal_data")
        mm_processor_kwargs = processed_inputs.get("mm_processor_kwargs")

        tokenized_length = len(prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id, length_penalty)

        beam_search_params = SamplingParams(
            logprobs=2 * beam_width,
            max_tokens=1,
            temperature=temperature,
        )
        all_beams = [
            BeamSearchSequence(tokens=prompt_token_ids,
                               cum_logprob=0,
                               logprobs=[],
                               multi_modal_data=multi_modal_data,
                               mm_processor_kwargs=mm_processor_kwargs)
        ]
        completed = []
        ###############################################
        start_time = time.time()

        for t in range(max_tokens):
            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens,
                             multi_modal_data=beam.multi_modal_data,
                             mm_processor_kwargs=beam.mm_processor_kwargs)
                for beam in all_beams
            ]

            tasks = []
            ###############################################测试beam_search的推理速度
            print(f"beam_search number {t + 1} token generate  start... ")
#             print(f"all_beams: {len(all_beams)}")
#             print(f"all_beams: {all_beams}")
            beam_start_time = time.time()
            
            
            request_id = f"beam_search-{random_uuid()}"
            for i, individual_prompt in enumerate(prompts_batch):
                request_id_item = f"{request_id}-{i}"
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.generate(individual_prompt, beam_search_params,
                                      request_id_item)))
                tasks.append(task)

            output = await asyncio.gather(*tasks)
            #############################################
            print(f"beam_search number {t + 1} token spend time: {time.time() - beam_start_time}")
            

            output = [x[0] for x in output]
#             print("output: ", output)
            new_beams = []
            
            ####################################
            concat_start_time = time.time()
            for i, current_beam in enumerate(all_beams):
                result = output[i]

                if result.outputs[0].logprobs is not None:
                    logprobs = result.outputs[0].logprobs[0]
                    for token_id, logprob_obj in logprobs.items():
                        if token_id == tokenizer.eos_token_id and \
                            not ignore_eos:
                            completed.append(
                                BeamSearchSequence(
                                    tokens=current_beam.tokens +
                                    [token_id] if include_stop_str_in_output
                                    else current_beam.tokens,
                                    logprobs=current_beam.logprobs +
                                    [logprobs],
                                    cum_logprob=current_beam.cum_logprob +
                                    logprob_obj.logprob,
                                    finish_reason="stop",
                                    stop_reason=tokenizer.eos_token_id))
                        else:
                            new_beams.append(
                                BeamSearchSequence(
                                    tokens=current_beam.tokens + [token_id],
                                    logprobs=current_beam.logprobs +
                                    [logprobs],
                                    cum_logprob=current_beam.cum_logprob +
                                    logprob_obj.logprob,
                                    multi_modal_data=current_beam.
                                    multi_modal_data,
                                    mm_processor_kwargs=current_beam.
                                    mm_processor_kwargs))
                            
            print(f"number {t + 1} token concat spend time: {time.time() - concat_start_time}")
            ###################################################
            sort_start_time = time.time()
            sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            print(f"number {t + 1} token sort spend time: {time.time() - sort_start_time}")
            all_beams = sorted_beams[:beam_width]

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]
        ###############################################################
        print(f"total spend: ",time.time() - start_time)

        for beam in best_beams:
            if (beam.tokens[-1] == tokenizer.eos_token_id and not ignore_eos):
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(text=beam.text,
                                 cumulative_logprob=beam.cum_logprob,
                                 token_ids=beam.tokens[tokenized_length:],
                                 index=i,
                                 logprobs=beam.logprobs,
                                 finish_reason=beam.finish_reason if
                                 beam.finish_reason is not None else "length",
                                 stop_reason=beam.stop_reason)
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None)

        yield beam_search_output

    async def beam_search_new(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        preprocessor = await self.get_input_preprocessor()
        tokenizer_group = preprocessor.get_tokenizer_group()
        tokenizer = await tokenizer_group.get_lora_tokenizer_async()

        if is_explicit_encoder_decoder_prompt(prompt):
            raise NotImplementedError
        else:
            processed_inputs = preprocessor._prompt_to_llm_inputs(prompt)

        prompt_token_ids = processed_inputs["prompt_token_ids"]
        prompt_text = processed_inputs.get("prompt")
        multi_modal_data = processed_inputs.get("multi_modal_data")
        mm_processor_kwargs = processed_inputs.get("mm_processor_kwargs")

        tokenized_length = len(prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id, length_penalty)

        logprobs_num = 2 * beam_width
        beam_search_params = SamplingParams(
            logprobs=logprobs_num,
            max_tokens=1,
            temperature=temperature,
        )
        all_beams = [
            BeamSearchSequence(tokens=prompt_token_ids,
                               cum_logprob=0,
                               logprobs=[],
                               multi_modal_data=multi_modal_data,
                               mm_processor_kwargs=mm_processor_kwargs)
        ]
        completed = []
        ###############################################
        start_time = time.time()

        for t in range(max_tokens):
            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens,
                             multi_modal_data=beam.multi_modal_data,
                             mm_processor_kwargs=beam.mm_processor_kwargs)
                for beam in all_beams
            ]

            tasks = []
            ###############################################测试beam_search的推理速度
            print(f"beam_search number {t + 1} token generate  start... ")
#             print(f"all_beams: {all_beams}")
            beam_start_time = time.time()
            
            
            request_id = f"beam_search-{random_uuid()}"
            for i, individual_prompt in enumerate(prompts_batch):
                request_id_item = f"{request_id}-{i}"
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.generate(individual_prompt, beam_search_params,
                                      request_id_item)))
                tasks.append(task)
            output = await asyncio.gather(*tasks)
            #############################################
            print(f"beam_search number {t + 1} token spend time: {time.time() - beam_start_time}")
            

            output = [x[0] for x in output]
#             print("output: ", output)
            new_beams = []
            
            ####################################
            """
                经测试结果拼接部分耗时也相对较长，优化双重循环为单循环，改为先排序后拼接的思路
                并且beam拼接前的累计概率以及prompt可以复用
            """
            """
            concat_start_time = time.time()
            for i, current_beam in enumerate(all_beams):
                result = output[i]

                if result.outputs[0].logprobs is not None:
                    logprobs = result.outputs[0].logprobs[0]
                    for token_id, logprob_obj in logprobs.items():
                        if token_id == tokenizer.eos_token_id and \
                            not ignore_eos:
                            completed.append(
                                BeamSearchSequence(
                                    tokens=current_beam.tokens +
                                    [token_id] if include_stop_str_in_output
                                    else current_beam.tokens,
                                    logprobs=current_beam.logprobs +
                                    [logprobs],
                                    cum_logprob=current_beam.cum_logprob +
                                    logprob_obj.logprob,
                                    finish_reason="stop",
                                    stop_reason=tokenizer.eos_token_id))
                        else:
                            new_beams.append(
                                BeamSearchSequence(
                                    tokens=current_beam.tokens + [token_id],
                                    logprobs=current_beam.logprobs +
                                    [logprobs],
                                    cum_logprob=current_beam.cum_logprob +
                                    logprob_obj.logprob,
                                    multi_modal_data=current_beam.
                                    multi_modal_data,
                                    mm_processor_kwargs=current_beam.
                                    mm_processor_kwargs))
                            
            print(f"number {t + 1} token concat spend time: {time.time() - concat_start_time}")
            ###################################################
            sort_start_time = time.time()
            sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            print(f"number {t + 1} token sort spend time: {time.time() - sort_start_time}")
            all_beams = sorted_beams[:beam_width]
            
            """
            concat_start_time = time.time()
            
            #存放所有beam新生成的token
            all_beams_token_id = []
            #存放所有beam生成新token后的累计概率
            all_beams_logprob = []
            #遍历所有beam当前token的推理结果
            for i, result in enumerate(output):
                current_beam = all_beams[i]
                if result.outputs[0].logprobs is not None:
                    # # {token_id -> logprob} for each sequence group.
                    # SampleLogprobs = list[dict[int, Logprob]]
                    logprobs = result.outputs[0].logprobs[0]
                    all_beams_token_id.extend(list(logprobs.keys()))
                    all_beams_logprob.extend([current_beam.cum_logprob + obj.logprob for obj in logprobs.values()])

            ##处理输出eos的token
            all_beams_token_id = np.array(all_beams_token_id)
            all_beams_logprob = np.array(all_beams_logprob)
#             print(type(all_beams_token_id[0]))
#             print(type(all_beams_logprob[0]))
#             print(all_beams_token_id)
#             print(all_beams_logprob)

            if not ignore_eos:
                #获取eos token在所有生成结果中的索引位置
                eos_idx = np.where(all_beams_token_id == tokenizer.eos_token_id)[0]
#                 print("eos_idx: ", eos_idx)
                for idx in eos_idx:
                    current_beam = all_beams[idx // logprobs_num]
                    result = output[idx // logprobs_num]
                    completed.append(
                        BeamSearchSequence(
                            tokens=current_beam.tokens +
                            [tokenizer.eos_token_id] if include_stop_str_in_output
                            else current_beam.tokens,
                            logprobs=current_beam.logprobs +
                            [result.outputs[0].logprobs[0]],
                            cum_logprob=float(all_beams_logprob[idx]),
                            finish_reason="stop",
                            stop_reason=tokenizer.eos_token_id))
                #处理结束后将eos的情况的logprob设置为负无穷
                all_beams_logprob[eos_idx] = -np.inf

            ######处理非eos token
            #获取beam_with个最大概率值的索引
            topn_idx = np.argpartition(-all_beams_logprob, beam_width)[:beam_width]
#             print("topn_idx: ",topn_idx)
            for idx in topn_idx:
                current_beam = all_beams[idx // logprobs_num]
                result = output[idx // logprobs_num]     
                token_id = int(all_beams_token_id[idx])
                new_beams.append(
                    BeamSearchSequence(
                        tokens=current_beam.tokens + [token_id],
                        logprobs=current_beam.logprobs +
                        [result.outputs[0].logprobs[0]],
                        cum_logprob=float(all_beams_logprob[idx]),
                        multi_modal_data=current_beam.
                        multi_modal_data,
                        mm_processor_kwargs=current_beam.
                        mm_processor_kwargs))
                            
            print(f"number {t + 1} token concat spend time: {time.time() - concat_start_time}")
            ###################################################
            # sort_start_time = time.time()
            # sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            # print(f"number {t + 1} token sort spend time: {time.time() - sort_start_time}")
            # all_beams = sorted_beams[:beam_width]

            all_beams = new_beams

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]
        ###############################################################
        print(f"total spend: ",time.time() - start_time)

        for beam in best_beams:
            if (beam.tokens[-1] == tokenizer.eos_token_id and not ignore_eos):
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(text=beam.text,
                                 cumulative_logprob=beam.cum_logprob,
                                 token_ids=beam.tokens[tokenized_length:],
                                 index=i,
                                 logprobs=beam.logprobs,
                                 finish_reason=beam.finish_reason if
                                 beam.finish_reason is not None else "length",
                                 stop_reason=beam.stop_reason)
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None)

        yield beam_search_output

    async def beam_search_greedy(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        preprocessor = await self.get_input_preprocessor()
        tokenizer_group = preprocessor.get_tokenizer_group()
        tokenizer = await tokenizer_group.get_lora_tokenizer_async()

        if is_explicit_encoder_decoder_prompt(prompt):
            raise NotImplementedError
        else:
            processed_inputs = preprocessor._prompt_to_llm_inputs(prompt)

        prompt_token_ids = processed_inputs["prompt_token_ids"]
        prompt_text = processed_inputs.get("prompt")
        multi_modal_data = processed_inputs.get("multi_modal_data")
        mm_processor_kwargs = processed_inputs.get("mm_processor_kwargs")

        tokenized_length = len(prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id, length_penalty)

        beam_search_params = SamplingParams(
            logprobs=2 * beam_width,
            max_tokens=1,
            temperature=temperature,
        )
        all_beams = [
            BeamSearchSequence(tokens=prompt_token_ids,
                               cum_logprob=0,
                               logprobs=[],
                               multi_modal_data=multi_modal_data,
                               mm_processor_kwargs=mm_processor_kwargs)
        ]
        completed = []
        ###############################################
        start_time = time.time()

        for t in range(max_tokens // 2):
            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens,
                             multi_modal_data=beam.multi_modal_data,
                             mm_processor_kwargs=beam.mm_processor_kwargs)
                for beam in all_beams
            ]

            tasks = []
            ###############################################测试beam_search的推理速度
            print(f"beam_search number {t + 1} token generate  start... ")
#             print(f"all_beams: {len(all_beams)}")
#             print(f"all_beams: {all_beams}")
            beam_start_time = time.time()
            
            request_id = f"beam_search-{random_uuid()}"
            for i, individual_prompt in enumerate(prompts_batch):
                request_id_item = f"{request_id}-{i}"
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.generate(individual_prompt, beam_search_params,
                                      request_id_item)))
                tasks.append(task)

            output = await asyncio.gather(*tasks)
            #############################################
            print(f"beam_search number {t + 1} token spend time: {time.time() - beam_start_time}")
            

            output = [x[0] for x in output]
#             print("output: ", output)
            new_beams = []
            
            ####################################
            concat_start_time = time.time()
            for i, current_beam in enumerate(all_beams):
                result = output[i]

                if result.outputs[0].logprobs is not None:
                    logprobs = result.outputs[0].logprobs[0]
                    for token_id, logprob_obj in logprobs.items():
                        if token_id == tokenizer.eos_token_id and \
                            not ignore_eos:
                            completed.append(
                                BeamSearchSequence(
                                    tokens=current_beam.tokens +
                                    [token_id] if include_stop_str_in_output
                                    else current_beam.tokens,
                                    logprobs=current_beam.logprobs +
                                    [logprobs],
                                    cum_logprob=current_beam.cum_logprob +
                                    logprob_obj.logprob,
                                    finish_reason="stop",
                                    stop_reason=tokenizer.eos_token_id))
                        else:
                            new_beams.append(
                                BeamSearchSequence(
                                    tokens=current_beam.tokens + [token_id],
                                    logprobs=current_beam.logprobs +
                                    [logprobs],
                                    cum_logprob=current_beam.cum_logprob +
                                    logprob_obj.logprob,
                                    multi_modal_data=current_beam.
                                    multi_modal_data,
                                    mm_processor_kwargs=current_beam.
                                    mm_processor_kwargs))
                            
            print(f"number {t + 1} token concat spend time: {time.time() - concat_start_time}")
            ###################################################
            sort_start_time = time.time()
            sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            print(f"number {t + 1} token sort spend time: {time.time() - sort_start_time}")
            all_beams = sorted_beams[:beam_width]

        # completed.extend(all_beams)
        # sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        # best_beams = sorted_completed[:beam_width]
        ###############################################################
        #####结束beam_search阶段，接下来将当前beam采用greedy search策略进行生成
        print(f"beam_search total spend: ",time.time() - start_time)

        prompts_batch = [
            TokensPrompt(prompt_token_ids=beam.tokens,
                            multi_modal_data=beam.multi_modal_data,
                            mm_processor_kwargs=beam.mm_processor_kwargs)
            for beam in all_beams
        ]

        tasks = []

        greedy_search_params = SamplingParams(
            logprobs=1,
            max_tokens=max_tokens - max_tokens // 2,
            temperature=0,
        )
        ###############################################测试greedy_search的推理速度
        print(f"greedy_search tokens generate  start... ")
#             print(f"all_beams: {len(all_beams)}")
#             print(f"all_beams: {all_beams}")
        greedy_start_time = time.time()
        
        
        req_id = f"greedy_search-{random_uuid()}"
        for i, individual_prompt in enumerate(prompts_batch):
            request_id_item = f"{req_id}-{i}"
            task = asyncio.create_task(
                collect_from_async_generator(
                    self.generate(individual_prompt, greedy_search_params,
                                    request_id_item)))
            tasks.append(task)

        output = await asyncio.gather(*tasks)
        #############################################
        print(f"greedy_search spend time: {time.time() - greedy_start_time}")
        

        output = [x[0] for x in output]
        new_beams = []
        ####################################
        for i, current_beam in enumerate(all_beams):
            outputs = output[i].outputs[0]
#             print("output: ", outputs)
            cumulative_logprob = sum([list(d.values())[0].logprob for d in outputs.logprobs])
#             print("cumulative_logprob: ", cumulative_logprob)
#             print("current_beam.cum_logprob: ", current_beam.cum_logprob)
            new_beams.append(
                BeamSearchSequence(
                    tokens=current_beam.tokens + outputs.token_ids,
                    logprobs=current_beam.logprobs +
                    outputs.logprobs,
                    cum_logprob=current_beam.cum_logprob +
                    cumulative_logprob,
                    finish_reason=outputs.finish_reason,
                    stop_reason=outputs.stop_reason,
                    multi_modal_data=current_beam.
                    multi_modal_data,
                    mm_processor_kwargs=current_beam.
                    mm_processor_kwargs))

        all_beams = new_beams
        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

        print(f"total spend: ",time.time() - start_time)

        for beam in best_beams:
            if (beam.tokens[-1] == tokenizer.eos_token_id and not ignore_eos):
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(text=beam.text,
                                 cumulative_logprob=beam.cum_logprob,
                                 token_ids=beam.tokens[tokenized_length:],
                                 index=i,
                                 logprobs=beam.logprobs,
                                 finish_reason=beam.finish_reason if
                                 beam.finish_reason is not None else "length",
                                 stop_reason=beam.stop_reason)
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None)

        yield beam_search_output        

    async def beam_search_inc(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        preprocessor = await self.get_input_preprocessor()
        tokenizer_group = preprocessor.get_tokenizer_group()
        tokenizer = await tokenizer_group.get_lora_tokenizer_async()

        if is_explicit_encoder_decoder_prompt(prompt):
            raise NotImplementedError
        else:
            processed_inputs = preprocessor._prompt_to_llm_inputs(prompt)

        prompt_token_ids = processed_inputs["prompt_token_ids"]
        prompt_text = processed_inputs.get("prompt")
        multi_modal_data = processed_inputs.get("multi_modal_data")
        mm_processor_kwargs = processed_inputs.get("mm_processor_kwargs")

        tokenized_length = len(prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id, length_penalty)


        #################################
        min_b = 2
        def exp_growth(step, max_steps, min_b, max_b):
            """确保最终步达到max_beam的指数增长"""
            if max_steps <= 1:
                return max_b
            
            if step >= max_steps - 1:
                return max_b
            
            growth_rate = (max_b / min_b) ** (1.0 / (max_steps - 1))
            current = min_b * (growth_rate ** step)
            return int(round(current))
        #######################################
        all_beams = [
            BeamSearchSequence(tokens=prompt_token_ids,
                               cum_logprob=0,
                               logprobs=[],
                               multi_modal_data=multi_modal_data,
                               mm_processor_kwargs=mm_processor_kwargs)
        ]
        completed = []
        ###############################################
        start_time = time.time()

        for t in range(max_tokens):
            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens,
                             multi_modal_data=beam.multi_modal_data,
                             mm_processor_kwargs=beam.mm_processor_kwargs)
                for beam in all_beams
            ]

            tasks = []
            ###############################################测试beam_search的推理速度
            print(f"beam_search number {t + 1} token generate  start... ")
#             print(f"all_beams: {all_beams}")
            beam_start_time = time.time()
            #########################################
            beam_num = exp_growth(t,max_tokens,min_b,beam_width)
            print("current beam_width: ",beam_num)
            logprobs_num = 2 * beam_num
            beam_search_params = SamplingParams(
                logprobs=logprobs_num,
                max_tokens=1,
                temperature=temperature,
            )
            
            request_id = f"beam_search-{random_uuid()}"
            for i, individual_prompt in enumerate(prompts_batch):
                request_id_item = f"{request_id}-{i}"
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.generate(individual_prompt, beam_search_params,
                                      request_id_item)))
                tasks.append(task)

            output = await asyncio.gather(*tasks)
            #############################################
            print(f"beam_search number {t + 1} token spend time: {time.time() - beam_start_time}")
            

            output = [x[0] for x in output]
#             print("output: ", output)
            new_beams = []
            
            ####################################
            concat_start_time = time.time()
            
            #存放所有beam新生成的token
            all_beams_token_id = []
            #存放所有beam生成新token后的累计概率
            all_beams_logprob = []
            #遍历所有beam当前token的推理结果
            for i, result in enumerate(output):
                current_beam = all_beams[i]
                if result.outputs[0].logprobs is not None:
                    # # {token_id -> logprob} for each sequence group.
                    # SampleLogprobs = list[dict[int, Logprob]]
                    logprobs = result.outputs[0].logprobs[0]
                    all_beams_token_id.extend(list(logprobs.keys()))
                    all_beams_logprob.extend([current_beam.cum_logprob + obj.logprob for obj in logprobs.values()])

            ##处理输出eos的token
            all_beams_token_id = np.array(all_beams_token_id)
            all_beams_logprob = np.array(all_beams_logprob)
#             print(type(all_beams_token_id[0]))
#             print(type(all_beams_logprob[0]))
#             print(all_beams_token_id)
#             print(all_beams_logprob)

            if not ignore_eos:
                #获取eos token在所有生成结果中的索引位置
                eos_idx = np.where(all_beams_token_id == tokenizer.eos_token_id)[0]
#                 print("eos_idx: ", eos_idx)
                for idx in eos_idx:
                    current_beam = all_beams[idx // logprobs_num]
                    result = output[idx // logprobs_num]
                    completed.append(
                        BeamSearchSequence(
                            tokens=current_beam.tokens +
                            [tokenizer.eos_token_id] if include_stop_str_in_output
                            else current_beam.tokens,
                            logprobs=current_beam.logprobs +
                            [result.outputs[0].logprobs[0]],
                            cum_logprob=float(all_beams_logprob[idx]),
                            finish_reason="stop",
                            stop_reason=tokenizer.eos_token_id))
                #处理结束后将eos的情况的logprob设置为负无穷
                all_beams_logprob[eos_idx] = -np.inf

            ######处理非eos token
            #获取beam_with个最大概率值的索引
            topn_idx = np.argpartition(-all_beams_logprob, beam_num)[:beam_num]
#             print("topn_idx: ",topn_idx)
            for idx in topn_idx:
                current_beam = all_beams[idx // logprobs_num]
                result = output[idx // logprobs_num]     
                token_id = int(all_beams_token_id[idx])
                new_beams.append(
                    BeamSearchSequence(
                        tokens=current_beam.tokens + [token_id],
                        logprobs=current_beam.logprobs +
                        [result.outputs[0].logprobs[0]],
                        cum_logprob=float(all_beams_logprob[idx]),
                        multi_modal_data=current_beam.
                        multi_modal_data,
                        mm_processor_kwargs=current_beam.
                        mm_processor_kwargs))
                            
            print(f"number {t + 1} token concat spend time: {time.time() - concat_start_time}")
            ###################################################
            # sort_start_time = time.time()
            # sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            # print(f"number {t + 1} token sort spend time: {time.time() - sort_start_time}")
            # all_beams = sorted_beams[:beam_width]

            all_beams = new_beams

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]
        ###############################################################
        print(f"total spend: ",time.time() - start_time)

        for beam in best_beams:
            if (beam.tokens[-1] == tokenizer.eos_token_id and not ignore_eos):
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(text=beam.text,
                                 cumulative_logprob=beam.cum_logprob,
                                 token_ids=beam.tokens[tokenized_length:],
                                 index=i,
                                 logprobs=beam.logprobs,
                                 finish_reason=beam.finish_reason if
                                 beam.finish_reason is not None else "length",
                                 stop_reason=beam.stop_reason)
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None)

        yield beam_search_output
        
    async def beam_search_topp(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        preprocessor = await self.get_input_preprocessor()
        tokenizer_group = preprocessor.get_tokenizer_group()
        tokenizer = await tokenizer_group.get_lora_tokenizer_async()

        if is_explicit_encoder_decoder_prompt(prompt):
            raise NotImplementedError
        else:
            processed_inputs = preprocessor._prompt_to_llm_inputs(prompt)

        prompt_token_ids = processed_inputs["prompt_token_ids"]
        prompt_text = processed_inputs.get("prompt")
        multi_modal_data = processed_inputs.get("multi_modal_data")
        mm_processor_kwargs = processed_inputs.get("mm_processor_kwargs")

        tokenized_length = len(prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id, length_penalty)

        logprobs_num = 2 * beam_width
        beam_search_params = SamplingParams(
            logprobs=logprobs_num,
            max_tokens=1,
            temperature=temperature,
        )
        ########################
        topp = 0.9
        ########################

        all_beams = [
            BeamSearchSequence(tokens=prompt_token_ids,
                               cum_logprob=0,
                               logprobs=[],
                               multi_modal_data=multi_modal_data,
                               mm_processor_kwargs=mm_processor_kwargs)
        ]
        completed = []
        ###############################################
        start_time = time.time()

        for t in range(max_tokens):
            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens,
                             multi_modal_data=beam.multi_modal_data,
                             mm_processor_kwargs=beam.mm_processor_kwargs)
                for beam in all_beams
            ]

            tasks = []
            ###############################################测试beam_search的推理速度
            print(f"beam_search number {t + 1} token generate  start... ")
#             print(f"all_beams: {all_beams}")
            beam_start_time = time.time()
            
            
            request_id = f"beam_search-{random_uuid()}"
            for i, individual_prompt in enumerate(prompts_batch):
                request_id_item = f"{request_id}-{i}"
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.generate(individual_prompt, beam_search_params,
                                      request_id_item)))
                tasks.append(task)

            output = await asyncio.gather(*tasks)
            #############################################
            print(f"beam_search number {t + 1} token spend time: {time.time() - beam_start_time}")
            

            output = [x[0] for x in output]
#             print("output: ", output)
            new_beams = []
            
            concat_start_time = time.time()
            
            #存放所有beam新生成的token
            all_beams_token_id = []
            #存放所有beam生成新token后的累计概率
            all_beams_logprob = []
            #遍历所有beam当前token的推理结果
            for i, result in enumerate(output):
                current_beam = all_beams[i]
                if result.outputs[0].logprobs is not None:
                    # # {token_id -> logprob} for each sequence group.
                    # SampleLogprobs = list[dict[int, Logprob]]
                    logprobs = result.outputs[0].logprobs[0]
                    all_beams_token_id.extend(list(logprobs.keys()))
                    all_beams_logprob.extend([current_beam.cum_logprob + obj.logprob for obj in logprobs.values()])

            ##处理输出eos的token
            all_beams_token_id = np.array(all_beams_token_id)
            all_beams_logprob = np.array(all_beams_logprob)
#             print(type(all_beams_token_id[0]))
#             print(type(all_beams_logprob[0]))
#             print(all_beams_token_id)
#             print(all_beams_logprob)

            if not ignore_eos:
                #获取eos token在所有生成结果中的索引位置
                eos_idx = np.where(all_beams_token_id == tokenizer.eos_token_id)[0]
#                 print("eos_idx: ", eos_idx)
                for idx in eos_idx:
                    current_beam = all_beams[idx // logprobs_num]
                    result = output[idx // logprobs_num]
                    completed.append(
                        BeamSearchSequence(
                            tokens=current_beam.tokens +
                            [tokenizer.eos_token_id] if include_stop_str_in_output
                            else current_beam.tokens,
                            logprobs=current_beam.logprobs +
                            [result.outputs[0].logprobs[0]],
                            cum_logprob=float(all_beams_logprob[idx]),
                            finish_reason="stop",
                            stop_reason=tokenizer.eos_token_id))
                #处理结束后将eos的情况的logprob设置为负无穷
                all_beams_logprob[eos_idx] = -np.inf

            ######处理非eos token
            #获取beam_with个最大概率值的索引
            partition_indices = np.argpartition(-all_beams_logprob, beam_width)[:beam_width]
            partition_values = all_beams_logprob[partition_indices]
            sorted_order = np.argsort(-partition_values)
            topn_idx = partition_indices[sorted_order]

            topn_probs = np.exp(all_beams_logprob[topn_idx])
#             print("topn_probs: ", topn_probs)
            norm_probs = topn_probs / np.sum(topn_probs)
#             print("norm_probs: ", norm_probs)
            probs_sum = 0

#             print("topn_idx: ",topn_idx)
            for i, idx in enumerate(topn_idx):
                current_beam = all_beams[idx // logprobs_num]
                result = output[idx // logprobs_num]     
                token_id = int(all_beams_token_id[idx])
                new_beams.append(
                    BeamSearchSequence(
                        tokens=current_beam.tokens + [token_id],
                        logprobs=current_beam.logprobs +
                        [result.outputs[0].logprobs[0]],
                        cum_logprob=float(all_beams_logprob[idx]),
                        multi_modal_data=current_beam.
                        multi_modal_data,
                        mm_processor_kwargs=current_beam.
                        mm_processor_kwargs))
                if t + 1 == max_tokens:
                    continue
                probs_sum = probs_sum + norm_probs[i]
                if probs_sum > topp:
                    break

            print("current beam_width: ",len(new_beams))
                            
            print(f"number {t + 1} token concat spend time: {time.time() - concat_start_time}")
            ###################################################
            # sort_start_time = time.time()
            # sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            # print(f"number {t + 1} token sort spend time: {time.time() - sort_start_time}")
            # all_beams = sorted_beams[:beam_width]

            all_beams = new_beams

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]
        ###############################################################
        print(f"total spend: ",time.time() - start_time)

        for beam in best_beams:
            if (beam.tokens[-1] == tokenizer.eos_token_id and not ignore_eos):
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(text=beam.text,
                                 cumulative_logprob=beam.cum_logprob,
                                 token_ids=beam.tokens[tokenized_length:],
                                 index=i,
                                 logprobs=beam.logprobs,
                                 finish_reason=beam.finish_reason if
                                 beam.finish_reason is not None else "length",
                                 stop_reason=beam.stop_reason)
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None)

        yield beam_search_output

        
        
    async def beam_search_sync(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        preprocessor = await self.get_input_preprocessor()
        tokenizer_group = preprocessor.get_tokenizer_group()
        tokenizer = await tokenizer_group.get_lora_tokenizer_async()

        if is_explicit_encoder_decoder_prompt(prompt):
            raise NotImplementedError
        else:
            processed_inputs = preprocessor._prompt_to_llm_inputs(prompt)

        prompt_token_ids = processed_inputs["prompt_token_ids"]
        prompt_text = processed_inputs.get("prompt")
        multi_modal_data = processed_inputs.get("multi_modal_data")
        mm_processor_kwargs = processed_inputs.get("mm_processor_kwargs")

        tokenized_length = len(prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id, length_penalty)

        logprobs_num = 2 * beam_width
        beam_search_params = SamplingParams(
            n=beam_width,
            logprobs=logprobs_num,
            max_tokens=1,
            temperature=temperature,
            output_kind=RequestOutputKind.FINAL_ONLY
        )
        all_beams = [
            BeamSearchSequence(tokens=prompt_token_ids,
                               cum_logprob=0,
                               logprobs=[],
                               multi_modal_data=multi_modal_data,
                               mm_processor_kwargs=mm_processor_kwargs)
        ]
        completed = []
        ###############################################
        start_time = time.time()

        for t in range(max_tokens):
            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens,
                             multi_modal_data=beam.multi_modal_data,
                             mm_processor_kwargs=beam.mm_processor_kwargs)
                for beam in all_beams
            ]
            beam_search_params.n = len(prompts_batch)
            ###############################################测试beam_search的推理速度
            print(f"beam_search number {t + 1} token generate  start... ")
#             print(f"all_beams: {all_beams}")
            beam_start_time = time.time()
            
            request_id = f"beam_search-{random_uuid()}"
            output = await collect_from_async_generator(self.generate_beam_search(
                prompts_batch, beam_search_params,request_id))

            #############################################
            print(f"beam_search number {t + 1} token spend time: {time.time() - beam_start_time}")
            

            output = [x for x in output[0].outputs]
#             print("output: ", output)
#             print("output lens: ", len(output))
            new_beams = []
            
            concat_start_time = time.time()
            
            #存放所有beam新生成的token
            all_beams_token_id = []
            #存放所有beam生成新token后的累计概率
            all_beams_logprob = []
            #遍历所有beam当前token的推理结果
            for i, result in enumerate(output):
                current_beam = all_beams[i]
                if result.logprobs is not None:
                    # # {token_id -> logprob} for each sequence group.
                    # SampleLogprobs = list[dict[int, Logprob]]
                    logprobs = result.logprobs[0]
                    all_beams_token_id.extend(list(logprobs.keys()))
                    all_beams_logprob.extend([current_beam.cum_logprob + obj.logprob for obj in logprobs.values()])

            ##处理输出eos的token
            all_beams_token_id = np.array(all_beams_token_id)
            all_beams_logprob = np.array(all_beams_logprob)
#             print(type(all_beams_token_id[0]))
#             print(type(all_beams_logprob[0]))
#             print(all_beams_token_id)
#             print(all_beams_logprob)

            if not ignore_eos:
                #获取eos token在所有生成结果中的索引位置
                eos_idx = np.where(all_beams_token_id == tokenizer.eos_token_id)[0]
#                 print("eos_idx: ", eos_idx)
                for idx in eos_idx:
                    current_beam = all_beams[idx // logprobs_num]
                    result = output[idx // logprobs_num]
                    completed.append(
                        BeamSearchSequence(
                            tokens=current_beam.tokens +
                            [tokenizer.eos_token_id] if include_stop_str_in_output
                            else current_beam.tokens,
                            logprobs=current_beam.logprobs +
                            [result.logprobs[0]],
                            cum_logprob=float(all_beams_logprob[idx]),
                            finish_reason="stop",
                            stop_reason=tokenizer.eos_token_id))
                #处理结束后将eos的情况的logprob设置为负无穷
                all_beams_logprob[eos_idx] = -np.inf

            ######处理非eos token
            #获取beam_with个最大概率值的索引
            topn_idx = np.argpartition(-all_beams_logprob, beam_width)[:beam_width]
#             print("topn_idx: ",topn_idx)
            for idx in topn_idx:
                current_beam = all_beams[idx // logprobs_num]
                result = output[idx // logprobs_num]     
                token_id = int(all_beams_token_id[idx])
                new_beams.append(
                    BeamSearchSequence(
                        tokens=current_beam.tokens + [token_id],
                        logprobs=current_beam.logprobs +
                        [result.logprobs[0]],
                        cum_logprob=float(all_beams_logprob[idx]),
                        multi_modal_data=current_beam.
                        multi_modal_data,
                        mm_processor_kwargs=current_beam.
                        mm_processor_kwargs))
                            
            print(f"number {t + 1} token concat spend time: {time.time() - concat_start_time}")
            ###################################################
            # sort_start_time = time.time()
            # sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            # print(f"number {t + 1} token sort spend time: {time.time() - sort_start_time}")
            # all_beams = sorted_beams[:beam_width]

            all_beams = new_beams

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]
        ###############################################################
        print(f"total spend: ",time.time() - start_time)

        for beam in best_beams:
            if (beam.tokens[-1] == tokenizer.eos_token_id and not ignore_eos):
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(text=beam.text,
                                 cumulative_logprob=beam.cum_logprob,
                                 token_ids=beam.tokens[tokenized_length:],
                                 index=i,
                                 logprobs=beam.logprobs,
                                 finish_reason=beam.finish_reason if
                                 beam.finish_reason is not None else "length",
                                 stop_reason=beam.stop_reason)
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None)

        yield beam_search_output        


    async def beam_search_post(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        preprocessor = await self.get_input_preprocessor()
        tokenizer_group = preprocessor.get_tokenizer_group()
        tokenizer = await tokenizer_group.get_lora_tokenizer_async()

        if is_explicit_encoder_decoder_prompt(prompt):
            raise NotImplementedError
        else:
            processed_inputs = preprocessor._prompt_to_llm_inputs(prompt)

        prompt_token_ids = processed_inputs["prompt_token_ids"]
        prompt_text = processed_inputs.get("prompt")
        multi_modal_data = processed_inputs.get("multi_modal_data")
        mm_processor_kwargs = processed_inputs.get("mm_processor_kwargs")

        tokenized_length = len(prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id, length_penalty)

        logprobs_num = 2 * beam_width
        beam_search_params = SamplingParams(
            n=beam_width,
            logprobs=logprobs_num,
            max_tokens=1,
            temperature=temperature,
            output_kind=RequestOutputKind.FINAL_ONLY
        )
        all_beams = [
            BeamSearchSequence(tokens=prompt_token_ids,
                               cum_logprob=0,
                               logprobs=[],
                               multi_modal_data=multi_modal_data,
                               mm_processor_kwargs=mm_processor_kwargs)
        ]
        completed = []
        ###############################################
        start_time = time.time()

        for t in range(max_tokens):
            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens,
                             multi_modal_data=beam.multi_modal_data,
                             mm_processor_kwargs=beam.mm_processor_kwargs)
                for beam in all_beams
            ]
            beam_search_params.n = len(prompts_batch)
            ###############################################测试beam_search的推理速度
            print(f"beam_search number {t + 1} token generate  start... ")
#             print(f"all_beams: {all_beams}")
            beam_start_time = time.time()
            
            request_id = f"beam_search-{random_uuid()}"
            output = await collect_from_async_generator(self.generate_beam_search_post(
                prompts_batch, beam_search_params,request_id))

            #############################################
            print(f"beam_search number {t + 1} token spend time: {time.time() - beam_start_time}")
            

            output = [x for x in output[0].outputs]
#             print("output: ", output)
#             print("output lens: ", len(output))
            new_beams = []
            
            concat_start_time = time.time()
            
            #存放所有beam新生成的token
            all_beams_token_id = []
            #存放所有beam生成新token后的累计概率
            all_beams_logprob = []
            #遍历所有beam当前token的推理结果
            for i, result in enumerate(output):
                current_beam = all_beams[i]
                if result.simple_logprobs is not None:
                    # # {token_id -> logprob} for each sequence group.
                    # SampleLogprobs = list[dict[int, Logprob]]
                    tokenids, logprobs = result.simple_logprobs[0]
#                     print("tokenids: ", tokenids)
#                     print("logprobs: ", logprobs)
                    all_beams_token_id.extend(tokenids[1:])
                    all_beams_logprob.extend([current_beam.cum_logprob + p for p in logprobs[1:]])

            ##处理输出eos的token
            all_beams_token_id = np.array(all_beams_token_id)
            all_beams_logprob = np.array(all_beams_logprob)
#             print(type(all_beams_token_id[0]))
#             print(type(all_beams_logprob[0]))
#             print(all_beams_token_id)
#             print(all_beams_logprob)

            if not ignore_eos:
                #获取eos token在所有生成结果中的索引位置
                eos_idx = np.where(all_beams_token_id == tokenizer.eos_token_id)[0]
#                 print("eos_idx: ", eos_idx)
                for idx in eos_idx:
                    current_beam = all_beams[idx // logprobs_num]
                    result = output[idx // logprobs_num]
                    completed.append(
                        BeamSearchSequence(
                            tokens=current_beam.tokens +
                            [tokenizer.eos_token_id] if include_stop_str_in_output
                            else current_beam.tokens,
                            logprobs=[],
                            cum_logprob=float(all_beams_logprob[idx]),
                            finish_reason="stop",
                            stop_reason=tokenizer.eos_token_id))
                #处理结束后将eos的情况的logprob设置为负无穷
                all_beams_logprob[eos_idx] = -np.inf

            ######处理非eos token
            #获取beam_with个最大概率值的索引
            topn_idx = np.argpartition(-all_beams_logprob, beam_width)[:beam_width]
#             print("topn_idx: ",topn_idx)
            for idx in topn_idx:
                current_beam = all_beams[idx // logprobs_num]
                result = output[idx // logprobs_num]     
                token_id = int(all_beams_token_id[idx])
                new_beams.append(
                    BeamSearchSequence(
                        tokens=current_beam.tokens + [token_id],
                        logprobs=[],
                        cum_logprob=float(all_beams_logprob[idx]),
                        multi_modal_data=current_beam.
                        multi_modal_data,
                        mm_processor_kwargs=current_beam.
                        mm_processor_kwargs))
                            
            print(f"number {t + 1} token concat spend time: {time.time() - concat_start_time}")
            ###################################################
            # sort_start_time = time.time()
            # sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            # print(f"number {t + 1} token sort spend time: {time.time() - sort_start_time}")
            # all_beams = sorted_beams[:beam_width]

            all_beams = new_beams

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]
        ###############################################################
        print(f"total spend: ",time.time() - start_time)

        for beam in best_beams:
            if (beam.tokens[-1] == tokenizer.eos_token_id and not ignore_eos):
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(text=beam.text,
                                 cumulative_logprob=beam.cum_logprob,
                                 token_ids=beam.tokens[tokenized_length:],
                                 index=i,
                                 logprobs=beam.logprobs,
                                 finish_reason=beam.finish_reason if
                                 beam.finish_reason is not None else "length",
                                 stop_reason=beam.stop_reason)
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None)

        yield beam_search_output        
        

    async def beam_search(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
    ) -> AsyncGenerator[RequestOutput, None]:

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        preprocessor = await self.get_input_preprocessor()
        tokenizer_group = preprocessor.get_tokenizer_group()
        tokenizer = await tokenizer_group.get_lora_tokenizer_async()

        if is_explicit_encoder_decoder_prompt(prompt):
            raise NotImplementedError
        else:
            processed_inputs = preprocessor._prompt_to_llm_inputs(prompt)

        prompt_token_ids = processed_inputs["prompt_token_ids"]
        prompt_text = processed_inputs.get("prompt")
        multi_modal_data = processed_inputs.get("multi_modal_data")
        mm_processor_kwargs = processed_inputs.get("mm_processor_kwargs")

        tokenized_length = len(prompt_token_ids)

        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id, length_penalty)

        
        #################################
        min_b = 2
        def exp_growth(step, max_steps, min_b, max_b):
            """确保最终步达到max_beam的指数增长"""
            if max_steps <= 1:
                return max_b
            
            if step >= max_steps - 1:
                return max_b
            
            growth_rate = (max_b / min_b) ** (1.0 / (max_steps - 1))
            current = min_b * (growth_rate ** step)
            return int(round(current))
        #######################################
        
        
        all_beams = [
            BeamSearchSequence(tokens=prompt_token_ids,
                               cum_logprob=0,
                               logprobs=[],
                               multi_modal_data=multi_modal_data,
                               mm_processor_kwargs=mm_processor_kwargs)
        ]
        completed = []
        ###############################################
        start_time = time.time()

        for t in range(max_tokens):
            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens,
                             multi_modal_data=beam.multi_modal_data,
                             mm_processor_kwargs=beam.mm_processor_kwargs)
                for beam in all_beams
            ]
            ###############################################测试beam_search的推理速度
            print(f"beam_search number {t + 1} token generate  start... ")
#             print(f"all_beams: {all_beams}")
            beam_start_time = time.time()

            beam_num = exp_growth(t,max_tokens,min_b,beam_width)
            print("current beam_width: ",beam_num)
            logprobs_num = 2 * beam_num
            
            beam_search_params = SamplingParams(
                logprobs=logprobs_num,
                max_tokens=1,
                temperature=temperature,
                output_kind=RequestOutputKind.FINAL_ONLY
            )
            beam_search_params.n = len(prompts_batch)
            
            
            request_id = f"beam_search-{random_uuid()}"
            output = await collect_from_async_generator(self.generate_beam_search_post(
                prompts_batch, beam_search_params,request_id))

            #############################################
            print(f"beam_search number {t + 1} token spend time: {time.time() - beam_start_time}")
            

            output = [x for x in output[0].outputs]
#             print("output: ", output)
#             print("output lens: ", len(output))
            new_beams = []
            
            concat_start_time = time.time()
            
            #存放所有beam新生成的token
            all_beams_token_id = []
            #存放所有beam生成新token后的累计概率
            all_beams_logprob = []
            #遍历所有beam当前token的推理结果
            for i, result in enumerate(output):
                current_beam = all_beams[i]
                if result.simple_logprobs is not None:
                    # # {token_id -> logprob} for each sequence group.
                    # SampleLogprobs = list[dict[int, Logprob]]
                    tokenids, logprobs = result.simple_logprobs[0]
#                     print("tokenids: ", tokenids)
#                     print("logprobs: ", logprobs)
                    all_beams_token_id.extend(tokenids[1:])
                    all_beams_logprob.extend([current_beam.cum_logprob + p for p in logprobs[1:]])

            ##处理输出eos的token
            all_beams_token_id = np.array(all_beams_token_id)
            all_beams_logprob = np.array(all_beams_logprob)
#             print(type(all_beams_token_id[0]))
#             print(type(all_beams_logprob[0]))
#             print(all_beams_token_id)
#             print(all_beams_logprob)

            if not ignore_eos:
                #获取eos token在所有生成结果中的索引位置
                eos_idx = np.where(all_beams_token_id == tokenizer.eos_token_id)[0]
#                 print("eos_idx: ", eos_idx)
                for idx in eos_idx:
                    current_beam = all_beams[idx // logprobs_num]
                    result = output[idx // logprobs_num]
                    completed.append(
                        BeamSearchSequence(
                            tokens=current_beam.tokens +
                            [tokenizer.eos_token_id] if include_stop_str_in_output
                            else current_beam.tokens,
                            logprobs=[],
                            cum_logprob=float(all_beams_logprob[idx]),
                            finish_reason="stop",
                            stop_reason=tokenizer.eos_token_id))
                #处理结束后将eos的情况的logprob设置为负无穷
                all_beams_logprob[eos_idx] = -np.inf

            ######处理非eos token
            #获取beam_with个最大概率值的索引
            topn_idx = np.argpartition(-all_beams_logprob, beam_num)[:beam_num]
#             print("topn_idx: ",topn_idx)
            for idx in topn_idx:
                current_beam = all_beams[idx // logprobs_num]
                result = output[idx // logprobs_num]     
                token_id = int(all_beams_token_id[idx])
                new_beams.append(
                    BeamSearchSequence(
                        tokens=current_beam.tokens + [token_id],
                        logprobs=[],
                        cum_logprob=float(all_beams_logprob[idx]),
                        multi_modal_data=current_beam.
                        multi_modal_data,
                        mm_processor_kwargs=current_beam.
                        mm_processor_kwargs))
                            
            print(f"number {t + 1} token concat spend time: {time.time() - concat_start_time}")
            ###################################################
            # sort_start_time = time.time()
            # sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            # print(f"number {t + 1} token sort spend time: {time.time() - sort_start_time}")
            # all_beams = sorted_beams[:beam_width]

            all_beams = new_beams

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]
        ###############################################################
        print(f"total spend: ",time.time() - start_time)

        for beam in best_beams:
            if (beam.tokens[-1] == tokenizer.eos_token_id and not ignore_eos):
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(text=beam.text,
                                 cumulative_logprob=beam.cum_logprob,
                                 token_ids=beam.tokens[tokenized_length:],
                                 index=i,
                                 logprobs=beam.logprobs,
                                 finish_reason=beam.finish_reason if
                                 beam.finish_reason is not None else "length",
                                 stop_reason=beam.stop_reason)
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None)

        yield beam_search_output       
        
    @abstractmethod
    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model."""
        ...

    @abstractmethod
    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request.
        """
        ...

    @abstractmethod
    async def get_vllm_config(self) -> VllmConfig:
        """Get the vllm configuration of the vLLM engine."""
        ...

    @abstractmethod
    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        ...

    @abstractmethod
    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""
        ...

    @abstractmethod
    async def get_input_preprocessor(self) -> InputPreprocessor:
        """Get the input processor of the vLLM engine."""
        ...

    @abstractmethod
    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        """Get the appropriate tokenizer for the request"""
        ...

    @abstractmethod
    async def is_tracing_enabled(self) -> bool:
        ...

    @abstractmethod
    async def do_log_stats(
        self,
        scheduler_outputs: Optional[SchedulerOutputs] = None,
        model_output: Optional[List[SamplerOutput]] = None,
    ) -> None:
        ...

    @abstractmethod
    async def check_health(self) -> None:
        """Raise if unhealthy"""
        ...

    @abstractmethod
    async def start_profile(self) -> None:
        """Start profiling the engine"""
        ...

    @abstractmethod
    async def stop_profile(self) -> None:
        """Start profiling the engine"""
        ...

    @abstractmethod
    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        """Reset the prefix cache"""
        ...

    @abstractmethod
    async def sleep(self, level: int = 1) -> None:
        """Sleep the engine"""
        ...

    @abstractmethod
    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        """Wake up the engine"""
        ...

    @abstractmethod
    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        ...

    @abstractmethod
    async def add_lora(self, lora_request: LoRARequest) -> None:
        """Load a new LoRA adapter into the engine for future requests."""
        ...
