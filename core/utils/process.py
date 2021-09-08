import pdb


def hide_tags(chars, tags, target_tags):
    revised_chars, revised_tags = [], []
    for cha, tag in zip(chars, tags):
        if tag == 'O' or tag == '[CLS]' or tag == '[SEP]':
            ele_tag = tag
        else:
            ele_tag = tag.split('-')[1]
        if ele_tag in target_tags:
            if tag.split('-')[0] == 'S' or tag.split('-')[0] == 'B':
                revised_chars.append('<' + ele_tag + '>')
                revised_tags.append('O')
        else:
            revised_chars.append(cha)
            revised_tags.append(tag)
    return revised_chars, revised_tags
