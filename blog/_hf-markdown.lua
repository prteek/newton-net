-- Hugging Face articles render inline maths only as \\( ... \\), not $...$ (display $$ ... $$ already works).
function Math(m)
  if m.mathtype == "InlineMath" then
    return pandoc.RawInline("markdown", "\\\\( " .. m.text .. " \\\\)")
  end
end
