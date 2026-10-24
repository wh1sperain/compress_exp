"""Explicit, auditable compatibility bridges; never patch author source on disk."""
def llava_forward_with_cache_position(self,input_ids=None,attention_mask=None,
        position_ids=None,past_key_values=None,inputs_embeds=None,labels=None,
        use_cache=None,output_attentions=None,output_hidden_states=None,
        images=None,image_sizes=None,return_dict=None,cache_position=None):
    """LLaVA 4.37 wrapper with the 4.39 cache_position forwarded unchanged.

The multimodal preparation and Llama call are the original wrapper's two
operations. Dropping cache_position would hide the incompatibility and could
change decode semantics, so the new argument is explicitly propagated.
"""
    from transformers import LlamaForCausalLM
    if inputs_embeds is None:
        (input_ids,position_ids,attention_mask,past_key_values,inputs_embeds,labels)=self.prepare_inputs_labels_for_multimodal(
            input_ids,position_ids,attention_mask,past_key_values,labels,images,image_sizes)
    return LlamaForCausalLM.forward(self,input_ids=input_ids,attention_mask=attention_mask,
        position_ids=position_ids,past_key_values=past_key_values,inputs_embeds=inputs_embeds,
        labels=labels,use_cache=use_cache,output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,return_dict=return_dict,cache_position=cache_position)
