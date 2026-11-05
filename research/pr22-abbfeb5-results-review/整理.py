import pathlib,sys,hashlib,json,shutil
root=pathlib.Path(sys.argv[1]).resolve();r=pathlib.Path(__file__).resolve().parent;folder=root/'07_PR22_去冗余转换与Dropout';name='pr22-abbfeb5-20261101-0iE77k.tar.gz';src=root/name;dst=folder/name;h='d835979c011de0237e7163063b8fba52e443c4a5059844016ae9d70d0abeeedb';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert folder.parent==root and dst.parent==folder and src.parent==root and sha(src)==h
for p,b in [(root/'索引.md',r/'原索引备份.md'),(root/'文件清单.json',r/'原文件清单备份.json')]:
 if p.read_bytes()!=b.read_bytes():raise ValueError('existing index changed')
if dst.exists() and sha(dst)!=h:raise FileExistsError(dst)
for n in ['简短分析.md','核验清单.json']:
 if (folder/n).exists() and (folder/n).read_bytes()!=(r/n).read_bytes():raise FileExistsError(folder/n)
d=json.loads((root/'文件清单.json').read_text(encoding='utf-8-sig'));rel=folder.name+'/'+name
if any(x['整理后路径']==rel for x in d['文件']):raise ValueError('entry already exists')
d['文件'].append({'实验日期':'2026-11-01','原文件名':name,'整理后路径':rel,'字节数':src.stat().st_size,'SHA256':h})
index=(root/'索引.md').read_text(encoding='utf-8-sig').rstrip()+'\n- 2026-11-01：[ '+name+' ]('+rel+')\n'
folder.mkdir(exist_ok=True)
if not dst.exists():
 with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
assert sha(dst)==h and sha(src)==h
for n in ['简短分析.md','核验清单.json']:
 if not (folder/n).exists():shutil.copyfile(r/n,folder/n)
(root/'索引.md').write_text(index,encoding='utf8');(root/'文件清单.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('OK: verified copy; original retained')
