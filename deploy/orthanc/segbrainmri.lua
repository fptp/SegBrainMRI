-- Orthanc : envoie automatiquement chaque nouvelle série FLAIR (lésions de SEP) et chaque
-- T1 3D non injectée (volumétrie) vers SegBrainMRI.
-- Les résultats renvoyés par SegBrainMRI (Manufacturer = "SegBrainMRI", SEG) sont ignorés,
-- ce qui évite toute boucle.

local TARGET = 'segbrainmri'   -- nom de la modalité déclarée dans "DicomModalities"
local FLAIR_PATTERNS = { 'flair', 'dark.?fluid', 't2.?fl', 'spcir' }
local T1_PATTERNS = { 'bravo', 'mprage', 'mp%-rage', 'spgr', 't1.?3d', '3d.?t1', 'tfe' }
local T1_EXCLUDE = { 'gado', '%f[%w]gd%f[%W]', '%f[%w]post%f[%W]', '%+ ?c', 'contrast', 'inj' }

local function matches(text, patterns)
   for _, pattern in ipairs(patterns) do
      if string.find(text, pattern) then return true end
   end
   return false
end

function OnStableSeries(seriesId, tags, metadata)
   local modality = string.upper(tags['Modality'] or '')
   if modality ~= 'MR' or (tags['Manufacturer'] or '') == 'SegBrainMRI' then
      return
   end
   local text = string.lower((tags['SeriesDescription'] or '') .. ' ' ..
                             (tags['ProtocolName'] or '') .. ' ' ..
                             (tags['SequenceName'] or ''))
   local wanted = matches(text, FLAIR_PATTERNS)
      or (matches(text, T1_PATTERNS) and not matches(text, T1_EXCLUDE))
   if wanted then
      print('SegBrainMRI : envoi de la serie ' .. seriesId)
      RestApiPost('/modalities/' .. TARGET .. '/store',
                  '{"Resources": ["' .. seriesId .. '"], "Asynchronous": true}')
   end
end
