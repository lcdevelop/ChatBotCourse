package com.shareditor.chatbotv1;

import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;

public class HttpServerInboundHandler extends SimpleChannelInboundHandler<FullHttpRequest> {
	
	@Override
	protected void messageReceived(ChannelHandlerContext ctx, FullHttpRequest msg) throws Exception {
		NettyHttpServletResponse res = new NettyHttpServletResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK);
		Action.doServlet(msg, res);
		ChannelFuture future = ctx.channel().writeAndFlush(res);
		future.addListener(ChannelFutureListener.CLOSE);
	}

}
