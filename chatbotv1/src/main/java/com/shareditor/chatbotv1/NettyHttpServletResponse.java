package com.shareditor.chatbotv1;

import io.netty.buffer.ByteBuf;
import io.netty.handler.codec.http.DefaultHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaders;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;

public class NettyHttpServletResponse extends DefaultHttpResponse implements FullHttpResponse {
	
	private ByteBuf content;

	public NettyHttpServletResponse(HttpVersion version, HttpResponseStatus status) {
		super(version, status);
	}

	public HttpHeaders trailingHeaders() {
		// TODO Auto-generated method stub
		return null;
	}
	
	public void setContent(ByteBuf buf) {
		this.content = buf;
	}

	public ByteBuf content() {
		return content;
	}

	public int refCnt() {
		// TODO Auto-generated method stub
		return 0;
	}

	public boolean release() {
		// TODO Auto-generated method stub
		return false;
	}

	public boolean release(int decrement) {
		// TODO Auto-generated method stub
		return false;
	}

	public FullHttpResponse copy(ByteBuf newContent) {
		// TODO Auto-generated method stub
		return null;
	}

	public FullHttpResponse copy() {
		// TODO Auto-generated method stub
		return null;
	}

	public FullHttpResponse retain(int increment) {
		// TODO Auto-generated method stub
		return null;
	}

	public FullHttpResponse retain() {
		// TODO Auto-generated method stub
		return null;
	}

	public FullHttpResponse touch() {
		// TODO Auto-generated method stub
		return null;
	}

	public FullHttpResponse touch(Object hint) {
		// TODO Auto-generated method stub
		return null;
	}

	public FullHttpResponse duplicate() {
		// TODO Auto-generated method stub
		return null;
	}

	public FullHttpResponse setProtocolVersion(HttpVersion version) {
		// TODO Auto-generated method stub
		return null;
	}

	public FullHttpResponse setStatus(HttpResponseStatus status) {
		// TODO Auto-generated method stub
		return null;
	}
	
}
