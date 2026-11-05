import pathlib,sys,hashlib,json,shutil
root=pathlib.Path(sys.argv[1]).resolve();r=pathlib.Path(__file__).resolve().parent;folder=root/'06_PR22_KV缓存预分配';name='pr22-ef2b051-20261101-Olwchs-with-comparison.tar.gz';src=root/name;dst=folder/name;h='1c96fb071eb4032c1d195e58368d10b616c1fe5cc11b134c3c52a32523f0a44a';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert folder.parent==root and dst.parent==folder and src.parent==root and sha(src)==h
for p,b in [(root/'索引.md',r/'原索引备份.md'),(root/'文件清单.json',r/'原文件清单备份.json'),(folder/'简短分析.md',r/'原简短分析备份.md'),(folder/'核验清单.json',r/'原核验清单备份.json')]:
 if p.read_bytes()!=b.read_bytes():raise ValueError('existing file changed')
if dst.exists() and sha(dst)!=h:raise FileExistsError(dst)
d=json.loads((root/'文件清单.json').read_text(encoding='utf-8-sig'));rel=folder.name+'/'+name
if any(x['整理后路径']==rel for x in d['文件']):raise ValueError('entry already exists')
d['文件'].append({'实验日期':'2026-11-01','原文件名':name,'整理后路径':rel,'字节数':src.stat().st_size,'SHA256':h})
index=(root/'索引.md').read_text(encoding='utf-8-sig').rstrip()+'\n- 2026-11-01：[ '+name+' ]('+rel+')\n'
if not dst.exists():
 with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
assert sha(dst)==h and sha(src)==h
for n in ['简短分析.md','核验清单.json']:shutil.copyfile(r/n,folder/n)
(root/'索引.md').write_text(index,encoding='utf8');(root/'文件清单.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('OK: verified copy; original retained')
