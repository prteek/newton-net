-- Hugging Face articles render inline maths only as \\( ... \\), not $...$ (display $$ ... $$ already works).
function Math(m)
  if m.mathtype == "InlineMath" then
    return pandoc.RawInline("markdown", "\\\\( " .. m.text .. " \\\\)")
  end
end

-- Figures saved in assets/ link to their copy on the Hub (assets-url in the front matter, uploaded by
-- `./publish.sh <message> blog`), so the post needs no uploads in the article editor.
local assets_url

function Meta(meta)
  if meta["assets-url"] then
    assets_url = pandoc.utils.stringify(meta["assets-url"]):gsub("/$", "")
  end
end

function Image(img)
  if assets_url and img.src:match("^assets/") then
    img.src = assets_url .. "/" .. img.src
    return img
  end
end

return {{Meta = Meta}, {Math = Math, Image = Image}}
