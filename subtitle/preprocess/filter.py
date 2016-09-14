# coding:utf-8
import sys
import re
import chardet

if __name__ == '__main__':
    #illegal=ur"([\u2000-\u2010]+)"
    illegal=ur"([\u0000-\u2010]+)"
    pattern_illegals = [re.compile(ur"([\u2000-\u2010]+)"), re.compile(ur"([\u0090-\u0099]+)")]
    filters = ["字幕", "时间轴:", "校对:", "翻译:", "后期:", "监制:"]
    filters.append("时间轴：")
    filters.append("校对：")
    filters.append("翻译：")
    filters.append("后期：")
    filters.append("监制：")
    filters.append("禁止用作任何商业盈利行为")
    filters.append("http")
    htmltagregex = re.compile(r'<[^>]+>',re.S)
    brace_regex = re.compile(r'\{.*\}',re.S)
    slash_regex = re.compile(r'\\\w',re.S)
    repeat_regex = re.compile(r'[-=]{10}',re.S)
    f = open("./corpus/all.out", "r")
    count=0
    while True:
        line = f.readline()
        if line:
            line = line.strip()

            # 编码识别，不是utf-8就过滤
            gb_content = ''
            try:
                gb_content = line.decode("utf-8")
            except Exception as e:
                sys.stderr.write("decode error:  ", line)
                continue

            # 中文识别，不是中文就过滤
            need_continue = False
            for pattern_illegal in pattern_illegals:
                match_illegal = pattern_illegal.findall(gb_content)
                if len(match_illegal) > 0:
                    sys.stderr.write("match_illegal error: %s\n" % line)
                    need_continue = True
                    break
            if need_continue:
                continue

            # 关键词过滤
            need_continue = False
            for filter in filters:
                try:
                    line.index(filter)
                    sys.stderr.write("filter keyword of %s %s\n" % (filter, line))
                    need_continue = True
                    break
                except:
                    pass
            if need_continue:
                continue

            # 去掉剧集信息
            if re.match('.*第.*季.*', line):
                sys.stderr.write("filter copora %s\n" % line)
                continue
            if re.match('.*第.*集.*', line):
                sys.stderr.write("filter copora %s\n" % line)
                continue
            if re.match('.*第.*帧.*', line):
                sys.stderr.write("filter copora %s\n" % line)
                continue

            # 去html标签
            line = htmltagregex.sub('',line)

            # 去花括号修饰
            line = brace_regex.sub('', line)

            # 去转义
            line = slash_regex.sub('', line)

            # 去重复
            new_line = repeat_regex.sub('', line)
            if len(new_line) != len(line):
                continue

            # 去特殊字符
            line = line.replace('-', '').strip()

            if len(line) > 0:
                sys.stdout.write("%s\n" % line)
            count+=1
        else:
            break
    f.close()
    pass
