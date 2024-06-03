import LangSegment
import os
import time
import re
import webbrowser
LangSegment.setfilters(["zh","en","ja"])

def openweb(url):
    time.sleep(3)
    webbrowser.open(url)

# 数字转为中文读法
def num_to_chinese(num):
    num_str = str(num)
    chinese_digits = "零一二三四五六七八九"
    units = ["", "十", "百", "千"]
    big_units = ["", "万", "亿", "兆"]
    result = ""
    zero_flag = False  # 标记是否需要加'零'
    part = []  # 存储每4位的数字
    
    # 将数字按每4位分组
    while num_str:
        part.append(num_str[-4:])
        num_str = num_str[:-4]
    
    for i in range(len(part)):
        part_str = ""
        part_zero_flag = False
        for j in range(len(part[i])):
            digit = int(part[i][j])
            if digit == 0:
                part_zero_flag = True
            else:
                if part_zero_flag or (zero_flag and i > 0 and not result.startswith(chinese_digits[0])):
                    part_str += chinese_digits[0]
                    zero_flag = False
                    part_zero_flag = False
                part_str += chinese_digits[digit] + units[len(part[i]) - j - 1]
        if part_str.endswith("零"):
            part_str = part_str[:-1]  # 去除尾部的'零'
        if part_str:
            zero_flag = True
        
        if i > 0 and not set(part[i]) <= {'0'}:  # 如果当前部分不全是0，则加上相应的大单位
            result = part_str + big_units[i] + result
        else:
            result = part_str + result
    
    # 处理输入为0的情况或者去掉开头的零
    result = result.lstrip(chinese_digits[0])
    if not result:
        return chinese_digits[0]
    
    return result

# 数字转为英文读法
def num_to_english(num):
    
    num_str = str(num)
    # English representations for numbers 0-9
    english_digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    units = ["", "ten", "hundred", "thousand"]
    big_units = ["", "thousand", "million", "billion", "trillion"]
    result = ""
    need_and = False  # Indicates whether 'and' needs to be added
    part = []  # Stores each group of 4 digits
    is_first_part = True  # Indicates if it is the first part for not adding 'and' at the beginning
    
    # Split the number into 3-digit groups
    while num_str:
        part.append(num_str[-3:])
        num_str = num_str[:-3]
    
    part.reverse()
    
    for i, p in enumerate(part):
        p_str = ""
        digit_len = len(p)
        if int(p) == 0 and i < len(part) - 1:
            continue
        
        hundreds_digit = int(p) // 100 if digit_len == 3 else None
        tens_digit = int(p) % 100 if digit_len >= 2 else int(p[0] if digit_len == 1 else p[1])
        
        # Process hundreds
        if hundreds_digit is not None and hundreds_digit != 0:
            p_str += english_digits[hundreds_digit] + " hundred"
            if tens_digit != 0:
                p_str += " and "
        
        # Process tens and ones
        if 10 < tens_digit < 20:  # Teens exception
            teen_map = {
                11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
                16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"
            }
            p_str += teen_map[tens_digit]
        else:
            tens_map = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
            tens_val = tens_digit // 10
            ones_val = tens_digit % 10
            if tens_val >= 2:
                p_str += tens_map[tens_val] + (" " + english_digits[ones_val] if ones_val != 0 else "")
            elif tens_digit != 0 and tens_val < 2:  # When tens_digit is in [1, 9]
                p_str += english_digits[tens_digit]
        
        if p_str and not is_first_part and need_and:
            result += " and "
        result += p_str
        if i < len(part) - 1 and int(p) != 0:
            result += " " + big_units[len(part) - i - 1] + ", "
        
        is_first_part = False
        if int(p) != 0:
            need_and = True
    
    return result.capitalize()


# 数字转为中英文读法
def num2text(text):
    numtext=['零','一','二','三','四','五','六','七','八','九']
    point='点'
    lang='zh'
    # 英文字符长度超过一半
    if len(" ".join(re.findall(r'\b([a-zA-Z]{3,})\b',text,re.I)))>=len(text)/2:
        lang='en'
        numtext=[' zero ',' one ',' two ',' three ',' four ',' five ',' six ',' seven ',' eight ',' nine ']
        point=' point '
        
    # 取出数字 number_list= [('1000200030004000.123', '1000200030004000', '123'), ('23425', '23425', '')]
    number_list=re.findall('((\d+)(?:\.(\d+))?)',text)
    #print(number_list)
    if len(number_list)>0:            
        #dc= ('1000200030004000.123', '1000200030004000', '123')
        for m,dc in enumerate(number_list):
            if len(dc[1])>16:
                continue
            int_text=num_to_chinese(dc[1]) if lang=='zh' else num_to_english(dc[1])
            if len(dc)==3 and dc[2]:
                int_text+=point+"".join([numtext[int(i)] for i in dc[2]])
            
            text=text.replace(dc[0],int_text)
    if lang=='zh':
        return text.replace('1','一').replace('2','二').replace('3','三').replace('4','四').replace('5','五').replace('6','六').replace('7','七').replace('8','八').replace('9','九').replace('0','零')
        
    return text.replace('1',' one ').replace('2',' two ').replace('3',' three ').replace('4',' four ').replace('5',' five ').replace('6',' six ').replace('7','seven').replace('8',' eight ').replace('9',' nine ').replace('0',' zero ')



# 切分中英文并转换数字
def split_text(text_list):
    result=[]
    for i,text in enumerate(text_list):
        text_list[i]=num2text(text)
        '''
        continue
        text=text.replace('[uv_break]','<en>[uv_break]</en>').replace('[laugh]','<en>[laugh]</en>')
        langlist=LangSegment.getTexts(text)
        length=len(langlist)
        for i,t in enumerate(langlist):
            # 当前是控制符，则插入到前一个            
            if len(result)>0 and re.match(r'^[\s\,\.]*?\[(uv_break|laugh)\][\s\,\.]*$',t['text']) is not None:
                result[-1]+=t['text']
            else:
                result.append(num2text(t['text'],t['lang']))
        '''
    print(f'{text_list=}')
    return text_list



# 获取../static/wavs目录中的所有文件和目录并清理wav
def ClearWav(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    if not files:
        return False, "wavs目录内无wav文件"

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"已删除文件: {file_path}")
            elif os.path.isdir(file_path):
                print(f"跳过文件夹: {file_path}")
        except Exception as e:
            print(f"文件删除错误 {file_path}, 报错信息: {e}")
            return False, str(e)
    return True, "所有wav文件已被删除."