_DEFAULT_SYSTEM_PROMPT = "\nDirectly provide the final answer (not the justification) between <answer> and </answer> tags"
def evaluate_model(
        df, 
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        model_name="openrouter/anthropic/claude-haiku-4.5",
        scorer=None
    ):
    from . import score_answer
    if scorer is None:
        scorer = score_answer
    
    from litlm import complete, extract_answer
    
    y=complete(df.prompt + system_prompt, model=model_name)
    df['y'] = y
    df['format_check']=df['y'].map(lambda x: '</answer>' in x)
    df['pred']=df.y.map(extract_answer)
    df['score']=df.apply(lambda x: scorer(x.pred, x),axis=1)
    df.results = lambda: df.groupby('task')[['format_check', 'score']].mean().reset_index()
    return df
        