package com.shareditor.chatbotv1;

import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.AtomicReader;
import org.apache.lucene.index.AtomicReaderContext;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.classic.QueryParser.Operator;
import org.apache.lucene.queryparser.flexible.standard.builders.PhraseQueryNodeBuilder;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.FieldCache;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.PriorityQueue;
import org.apache.lucene.util.Version;
import org.wltea.analyzer.lucene.IKAnalyzer;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;

public class Action {
	
	private static final int MAX_RESULT = 20;
	private static final int MAX_TOTAL_HITS = 1000000;
	private static IndexSearcher indexSearcher = null;
	
	static {
		IndexReader reader = null;
		try {
			reader = DirectoryReader.open(FSDirectory.open(new File("/Users/lichuang/Developer/ChatBotCourse/chatbotv1/index")));
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		indexSearcher = new IndexSearcher(reader);
	}
	
	public static void doServlet(FullHttpRequest req, NettyHttpServletResponse res) throws IOException, ParseException {
		ByteBuf buf = null;
		QueryStringDecoder qsd = new QueryStringDecoder(req.uri());
		Map<String, List<String>> mapParameters = qsd.parameters();
		List<String> l = mapParameters.get("q");
		if (null != l && l.size() == 1) {
			String q = l.get(0);
			Query query = null;
			PriorityQueue<ScoreDoc> pq = new PriorityQueue<ScoreDoc>(MAX_RESULT) {
				
				@Override
				protected boolean lessThan(ScoreDoc a, ScoreDoc b) {
					return a.score < b.score;
				}
			};
			MyCollector collector = new MyCollector(pq);
			
			JSONObject ret = new JSONObject();
			TopDocs topDocs = collector.topDocs();
			
			Analyzer analyzer = new IKAnalyzer(true);
			QueryParser qp = new QueryParser(Version.LUCENE_4_9, "question", analyzer);
			if (topDocs.totalHits == 0) {
				qp.setDefaultOperator(Operator.AND);
				query = qp.parse(q);
				System.out.println(query.toString());
				indexSearcher.search(query, collector);
				topDocs = collector.topDocs();
			}
			
			if (topDocs.totalHits == 0) {
				qp.setDefaultOperator(Operator.OR);
				query = qp.parse(q);
				System.out.println(query.toString());
				indexSearcher.search(query, collector);
				topDocs = collector.topDocs();
			}
			
			
			ret.put("total", topDocs.totalHits);
			ret.put("q", q);
			JSONArray result = new JSONArray();
			for (ScoreDoc d : topDocs.scoreDocs) {
				Document doc = indexSearcher.doc(d.doc);
				String question = doc.get("question");
				String answer = doc.get("answer");
				JSONObject item = new JSONObject();
				item.put("question", question);
				item.put("answer", answer);
				item.put("score", d.score);
				item.put("doc", d.doc);
				result.add(item);
			}
			ret.put("result", result);
			buf = Unpooled.copiedBuffer(ret.toJSONString().getBytes());
		} else {
			buf = Unpooled.copiedBuffer("error".getBytes());
		}
		res.setContent(buf);
	}
	
	public static class MyCollector extends TopDocsCollector<ScoreDoc> {

		protected Scorer scorer;
		protected AtomicReader reader;
		protected int baseDoc;
		protected HashSet<Integer> set = new HashSet<Integer>();

		protected MyCollector(PriorityQueue<ScoreDoc> pq) {
			super(pq);
		}

		@Override
		public void setScorer(Scorer scorer) throws IOException {
			this.scorer = scorer;
		}

		@Override
		public void collect(int doc) throws IOException {
			
			if (this.totalHits > MAX_TOTAL_HITS) {
				return;
			}
			
			String answer = this.getAnswer(doc);
			if (set.contains(answer.hashCode())) {
				return;
			} else {
				set.add(answer.hashCode());
				ScoreDoc sd = new ScoreDoc(doc, this.scorer.score());
				if (this.pq.size() >= MAX_RESULT) {
					this.pq.updateTop();
					this.pq.pop();
				}
				this.pq.add(sd);
				this.totalHits++;
			}
		}
		
		@Override
		public void setNextReader(AtomicReaderContext context) throws IOException {
			this.reader = context.reader();
			this.baseDoc = context.docBase;
		}

		@Override
		public boolean acceptsDocsOutOfOrder() {
			return false;
		}
		
		private String getAnswer(int doc) throws IOException {
			Document d = indexSearcher.doc(doc);
			return d.get("answer");
		}
	}
}
