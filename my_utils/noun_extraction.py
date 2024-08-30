def noun_extraction(text, nlp, need_np=False):
    doc = nlp(text)

    chunks = {i: chunk for chunk in doc.noun_chunks for i in range(chunk.start, chunk.end)}

    head = next((token.head for token in doc if token.head.i == token.i), None)

    if head is None:
        if need_np:
            return text, text
        return text

    while head.i not in chunks:
        children = list(head.children)
        if children and children[0].i in chunks:
            head = children[0]
        else:
            if need_np:
                return text, text
            else:
                return text

    head_chunk = chunks[head.i]
    head_noun = head_chunk.root.text

    if need_np:
        return head_noun, head_chunk.text
    else:
        return head_noun