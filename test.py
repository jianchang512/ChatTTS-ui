import re

def remove_brackets(text):
    # 正则表达式，匹配不是特定组合的 [ ] 对
    text=re.sub(r'\[(uv_break|laugh|lbreak|break)\]',r' \1 ',text,re.I|re.S|re.M)
    print(text)
    # 使用 re.sub 替换掉匹配到的非特定组合的 [ ] 对
    newt=re.sub(r'\[|\]|！', '', text)
    return    re.sub(r'\s(uv_break|laugh|lbreak|break)(?=\s|$)', r' [\1] ', newt)

    

# 示例文本
text = "这是一个测试文本[uv_break]里面包含了一些[laugh]和[lbreak]以及[break] 符号，但还有一些其他的 [符号] 需要被删除。"

# 调用函数
cleaned_text = remove_brackets(text)
print(cleaned_text)