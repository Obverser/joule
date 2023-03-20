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

# Ensure gen folder is created
if !Dir.exists? "gen"
    Dir.mkdir "gen"
    Dir.mkdir "gen/posts"
end

# Serve HTML content
server = HTTP::Server.new do |context|
    context.response.content_type = type

    if context.request.path == "/" && context.request.method == "GET"
        File.open "./banner.html" do |file|
            IO.copy file, context.response
        end
    elsif context.request.path == "/github"
        if !File.exists? "./gen/github.html"
            GitHub.regenerate
        end

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
        Dir.open "./md" do |folder|
            folder.each_child do |md|
                File.open folder.path + "/" + md do |file|
                    if file.info.modification_time > blog_gen
                        blog_gen = Time.utc
                        puts "[SERVER] " + blog_gen.to_s + ": Regenerating blog posts"
        
                        Blog.regenerate

                        break
                    end
                end
            end
        end

        File.open "./gen/blog.html" do |file|
            IO.copy file, context.response
        end
    elsif context.request.path.starts_with? "/style/"
        style = context.request.path.lchop("/blog/").lchop("/style/")
        
        if style.starts_with? "SF"
            context.response.headers["Access-Control-Allow-Origin"] = "*"
            context.response.content_type = "font/otf"
            File.open "./style/" + style + ".otf" do |ttf|
                IO.copy ttf, context.response
            end
        elsif style == "YujiCafe"
            context.response.headers["Access-Control-Allow-Origin"] = "*"
            context.response.content_type = "font/tff"
            File.open "./style/YujiCafe.ttf" do |ttf|
                IO.copy ttf, context.response
            end
        elsif style == "base"
            context.response.content_type = "text/css"
            File.open "./style/base.css" do |css|
                IO.copy css, context.response
            end
        end
    elsif context.request.path.starts_with? "/blog/" 
        File.open "./gen/posts/" + context.request.path.lchop("/blog/") + ".html" do |html|
            IO.copy html, context.response
        end
    elsif context.request.path.starts_with? "/image/"
        context.response.headers["Access-Control-Allow-Origin"] = "*"
        context.response.content_type = "image/svg+xml"
        File.open "./resources/" + context.request.path.lchop("/image/") do |image|
            IO.copy image, context.response
        end
    else
        # Unknown path
    end
end

puts "[ INIT ] " + Time.utc.to_s + ": Listening: #{host}:#{port}"
server.bind_tcp host, port
server.listen