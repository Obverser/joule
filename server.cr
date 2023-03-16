require "http/server"
require "http/client"
require "./github.cr"
require "./blog.cr"

# Server information
port = 8080
host = "127.0.0.1"
type = "text/html"

# Generation parameters, arbitrary tracking times that will change on first request
github_gen = Time.utc(2016, 1, 1, 12, 12, 12)
blog_gen = Time.utc(2016, 1, 1, 12, 12, 12)

# Serve HTML content
server = HTTP::Server.new do |context|
    context.response.content_type = type

    if context.request.path == "/" && context.request.method == "GET"
        File.open "./banner.html" do |file|
            IO.copy file, context.response
        end
    elsif context.request.path == "/github"
        File.open "./gen/github.html" do |file|
            # Ensure that our GitHub page isn't stale
            if file.info.modification_time - github_gen >= 6.hour
                github_gen = Time.utc
                puts "[SERVER] " + github_gen.to_s + ": Regenerating GitHub static file"

                GitHub.regenerate
            end

            IO.copy file, context.response
        end
    elsif context.request.path == "/blog"
        File.open "./md" do |folder|
            # Ensure the blog posts are updated
            if folder.info.modification_time != blog_gen
                blog_gen = folder.info.modification_time
                puts "[SERVER] " + blog_gen.to_s + ": Regenerating blog posts"

                Blog.regenerate
            end
        end

        File.open "./gen/blog.html" do |file|
            IO.copy file, context.response
        end
    elsif context.request.path.starts_with? "/blog/"
        filepath = context.request.path.lchop "/blog/"
    elsif context.request.path.starts_with? "/font/"
        font = context.request.path.lchop "/font/"
        
        # We don't need to check this, we can just use File.open, but we don't want an attack vector
        if font == "Symbols.ttf"
            File.open "./font/Symbols.ttf" do |ttf|
                IO.copy ttf, context.response
            end
        elsif font == "YujiCafe.ttf"
            File.open "./font/YujiCafe.ttf" do |ttf|
                IO.copy ttf, context.response
            end
        end
    else
        # Unknown path
    end
end

puts "[ INIT ] " + Time.utc.to_s + ": Listening: #{host}:#{port}"
server.bind_tcp host, port
server.listen